"""
Compare YOLO-SEA (best.pt, scale s, SESA+BiFPN) avec des baselines YOLO26
pretrain COCO (s et large) : params, FLOPs, vitesse d'inférence, mAP/P/R.

Les baselines COCO n'ont pas de classe "navire" mais ont une classe "boat" :
on évalue leurs détections "boat" contre le ground truth navires (mono-
classe, puisque "combined" n'a de toute façon qu'une seule classe réelle,
"unknown"). Le filtrage se fait uniquement sur les prédictions (via
model.predict(classes=[boat_id])) — l'argument `classes` de model.val()
filtre aussi le ground truth côté Ultralytics, ce qui viderait le GT ici
puisqu'aucune boîte n'a la classe COCO "boat" (8) ; le matching IoU est donc
refait à la main avec les fonctions internes d'Ultralytics (box_iou,
match_predictions, ap_per_class) pour rester cohérent avec leur calcul de mAP.

Chaque modèle tourne dans son propre sous-processus (--only), relancé via
subprocess par le process principal : sur un GPU partagé avec le bureau
(VRAM limitée), c'est la seule façon de garantir que la mémoire d'un modèle
est entièrement rendue avant de charger le suivant (del + empty_cache() ne
suffisent pas, des références restent vivantes côté Python/validator).

Usage :
    python yolo_sea/eval_compare.py
    python yolo_sea/eval_compare.py --weights best.pt --dataset combined
    python yolo_sea/eval_compare.py --imgsz 640 --n-speed-samples 100
    python yolo_sea/eval_compare.py --limit 150        # test rapide/prudent sur 150 images
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Doit être positionné avant l'init du contexte CUDA (donc avant `import torch`) :
# réduit la fragmentation mémoire qui a fait planter YOLO26l en plein milieu d'un
# val() à batch=8 sur ce GPU laptop 8 Go (cf. message torch.OutOfMemoryError).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
IOUV = torch.linspace(0.5, 0.95, 10)
MODELS = [
    ("YOLO-SEA (s)", "sea"),
    ("YOLO-SEA (s) INT8", "sea_int8"),
    ("YOLO26s (COCO)", "yolo26s"),
    ("YOLO26l (COCO)", "yolo26l"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "best.pt"),
                   help="Poids YOLO-SEA à évaluer (défaut : best.pt à la racine)")
    p.add_argument("--dataset", default="combined",
                   help="Nom du dataset (clé dans datasets/<dataset>/data.yaml)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size (défaut : 8, volontairement bas — ce GPU laptop fait aussi "
                        "tourner le bureau, un batch trop gros peut figer l'affichage)")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-speed-samples", type=int, default=100,
                   help="Nb d'images de test utilisées pour mesurer la vitesse d'inférence")
    p.add_argument("--limit", type=int, default=None,
                   help="Limite le split test (val() inclus) aux N premières images, via un "
                        "data.yaml temporaire — utile pour un test rapide/prudent avant le run complet")
    p.add_argument("--only", choices=["sea", "sea_int8", "yolo26s", "yolo26l"], default=None,
                   help="Usage interne : évalue un seul modèle dans ce process et imprime son "
                        "résultat en JSON (utilisé par le process parent via subprocess)")
    return p.parse_args()


def build_subset_dataset(data_yaml, split, labels_dir, images, limit):
    """data.yaml temporaire pointant sur les N premières images (+ labels) de `images`."""
    import tempfile

    from ultralytics.utils import YAML

    subset = images[:limit]
    tmp_root = Path(tempfile.mkdtemp(prefix="eval_compare_subset_"))
    img_dir = tmp_root / "images" / split
    lbl_dir = tmp_root / "labels" / split
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for img in subset:
        (img_dir / img.name).symlink_to(img.resolve())
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            (lbl_dir / label.name).symlink_to(label.resolve())

    (tmp_root / "images" / "val").symlink_to(img_dir.resolve())  # check_det_dataset exige 'val'

    orig = YAML.load(data_yaml)
    YAML.save(tmp_root / "data.yaml", {
        "train": f"images/{split}", "val": "images/val", split: f"images/{split}",
        "names": orig["names"],
    })
    return tmp_root / "data.yaml", subset


def resolve_device(arg):
    if arg != "auto":
        return arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def model_stats(model, imgsz):
    """Params (M) et FLOPs (G)."""
    net = model.model
    n_params = sum(p.numel() for p in net.parameters())
    return n_params / 1e6, get_flops(net, imgsz)


def measure_speed(model, images, imgsz, device, warmup=5):
    """Vitesse d'inférence moyenne (ms/img), forward only, sur une liste de chemins d'images."""
    for img in images[:warmup]:
        model.predict(img, imgsz=imgsz, device=device, verbose=False)

    t0 = time.perf_counter()
    for img in images:
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - t0
    return (elapsed / len(images)) * 1000  # ms/img


def match_predictions(pred_cls, true_cls, iou):
    """Reprend ultralytics.engine.validator.BaseValidator.match_predictions (mêmes seuils IOUV)."""
    correct = np.zeros((pred_cls.shape[0], IOUV.shape[0])).astype(bool)
    correct_class = true_cls[:, None] == pred_cls
    iou = (iou * correct_class).numpy()
    for i, threshold in enumerate(IOUV.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


def read_gt_boxes_xyxy(label_path, img_w, img_h):
    """Boîtes ground truth d'un fichier label YOLO, en pixels xyxy (classe ignorée, mono-classe)."""
    if not label_path.exists():
        return torch.zeros((0, 4))
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, xc, yc, w, h = (float(v) for v in line.split()[:5])
        boxes.append([
            (xc - w / 2) * img_w, (yc - h / 2) * img_h,
            (xc + w / 2) * img_w, (yc + h / 2) * img_h,
        ])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))


def predict_with_retry(model, img_path, imgsz, device, classes, attempts=3):
    """Une image à la fois (pas de batch) avec retry sur OOM CUDA.

    Sur ce GPU laptop, la VRAM dispo dépend de la charge du bureau au moment T
    (Xorg/gnome-shell tournent sur le même GPU) — un batch qui passait il y a
    une minute peut OOM maintenant. batch=1 maximise la marge ; le retry gère
    le cas où même ça échoue ponctuellement (pic de charge du bureau).
    """
    for attempt in range(attempts):
        try:
            return model.predict(str(img_path), imgsz=imgsz, device=device,
                                  classes=classes, verbose=False)[0]
        except torch.OutOfMemoryError:
            if device == "cpu" or attempt == attempts - 1:
                raise
            torch.cuda.empty_cache()
            time.sleep(2)
    raise RuntimeError("unreachable")


def eval_single_class(model, target_cls_id, image_paths, labels_dir, imgsz, device):
    """mAP50 / mAP50-95 / P / R mono-classe : prédictions filtrées sur target_cls_id vs GT (toute classe)."""
    stats_tp, stats_conf = [], []
    n_gt_total = 0

    for i, img_path in enumerate(image_paths):
        try:
            result = predict_with_retry(model, img_path, imgsz, device, [target_cls_id])
        except torch.OutOfMemoryError:
            raise RuntimeError(
                f"OOM CUDA persistant sur {img_path.name} après plusieurs tentatives — "
                "la VRAM est probablement saturée par d'autres applis (navigateur, vidéo). "
                "Ferme-les et relance, ou réduis --imgsz."
            )

        if device != "cpu" and i % 50 == 0:
            torch.cuda.empty_cache()

        img_h, img_w = result.orig_shape
        label_path = labels_dir / (Path(result.path).stem + ".txt")
        gt_boxes = read_gt_boxes_xyxy(label_path, img_w, img_h)
        n_gt_total += gt_boxes.shape[0]

        pred_boxes = result.boxes.xyxy.cpu() if result.boxes is not None else torch.zeros((0, 4))
        pred_conf = result.boxes.conf.cpu() if result.boxes is not None else torch.zeros((0,))
        n_pred = pred_boxes.shape[0]

        if n_pred == 0:
            continue
        if gt_boxes.shape[0] == 0:
            tp = torch.zeros((n_pred, IOUV.shape[0]), dtype=torch.bool)
        else:
            iou = box_iou(gt_boxes, pred_boxes)
            tp = match_predictions(torch.zeros(n_pred), torch.zeros(gt_boxes.shape[0]), iou)

        stats_tp.append(tp.numpy())
        stats_conf.append(pred_conf.numpy())

    if not stats_tp or n_gt_total == 0:
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}

    tp = np.concatenate(stats_tp, 0)
    conf = np.concatenate(stats_conf, 0)
    pred_cls = np.zeros(tp.shape[0])
    target_cls = np.zeros(n_gt_total)

    _, _, p, r, _, ap, *_ = ap_per_class(tp, conf, pred_cls, target_cls)
    return {
        "map50": float(ap[:, 0].mean()) if len(ap) else 0.0,
        "map50_95": float(ap.mean()) if len(ap) else 0.0,
        "precision": float(p.mean()) if len(p) else 0.0,
        "recall": float(r.mean()) if len(r) else 0.0,
    }


def run_sea_val(weights, params_m, flops_g, speed_ms, args, data_yaml, device):
    """val() complet (multi-classe, contre le vrai ground truth) — partagé par SEA fp32 et INT8."""
    model = YOLO(weights, task="detect")
    val_results = model.val(
        data=str(data_yaml), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=device, plots=False, verbose=False,
    )
    rd = val_results.results_dict
    return {
        "params_m": params_m, "flops_g": flops_g, "speed_ms": speed_ms,
        "map50": rd.get("metrics/mAP50(B)"),
        "map50_95": rd.get("metrics/mAP50-95(B)"),
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
    }


def run_one(only, args, data_yaml, test_images, labels_dir, speed_images, device):
    """Évalue un seul modèle (appelé dans un sous-process isolé) et retourne sa ligne de résultats."""
    if only == "sea":
        print(f"\n{'='*70}\nYOLO-SEA : {args.weights}\n{'='*70}")
        model = YOLO(args.weights)
        params_m, flops_g = model_stats(model, args.imgsz)
        speed_ms = measure_speed(model, speed_images, args.imgsz, device)
        return run_sea_val(args.weights, params_m, flops_g, speed_ms, args, data_yaml, device)
    elif only == "sea_int8":
        # Même archi que YOLO-SEA fp32 (params/FLOPs identiques) : on les récupère depuis
        # best.pt, et on évalue les poids quantifiés via onnxruntime, forcé sur CPU — c'est
        # le seul provider installé ici (pas de onnxruntime-gpu), et l'INT8 ONNX standard
        # cible justement l'inférence CPU/edge.
        int8_path = str(ROOT / "best_int8.onnx")
        print(f"\n{'='*70}\nYOLO-SEA INT8 (ONNX) : {int8_path}\n{'='*70}")
        fp32_model = YOLO(args.weights)
        params_m, flops_g = model_stats(fp32_model, args.imgsz)
        del fp32_model
        model = YOLO(int8_path, task="detect")
        speed_ms = measure_speed(model, speed_images, args.imgsz, "cpu")
        return run_sea_val(int8_path, params_m, flops_g, speed_ms, args, data_yaml, "cpu")
    else:
        weight_name = f"{only}.pt"
        print(f"\n{'='*70}\n{only} (COCO) : {weight_name}\n{'='*70}")
        model = YOLO(weight_name)  # auto-download si absent
        boat_id = next(k for k, v in model.names.items() if v == "boat")
        params_m, flops_g = model_stats(model, args.imgsz)
        speed_ms = measure_speed(model, speed_images, args.imgsz, device)
        if device != "cpu":
            torch.cuda.empty_cache()
        metrics = eval_single_class(model, boat_id, test_images, labels_dir, args.imgsz, device)

    return {"params_m": params_m, "flops_g": flops_g, "speed_ms": speed_ms, **metrics}


RESULT_MARKER = "EVAL_COMPARE_RESULT_JSON:"


def main():
    args = parse_args()
    device = resolve_device(args.device)

    data_yaml = ROOT / "datasets" / args.dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml introuvable : {data_yaml}")

    images_dir = data_yaml.parent / "images" / args.split
    labels_dir = data_yaml.parent / "labels" / args.split
    test_images = sorted(images_dir.glob("*"))
    if not test_images:
        raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")

    if args.limit:
        data_yaml, test_images = build_subset_dataset(
            data_yaml, args.split, labels_dir, test_images, args.limit)
        labels_dir = data_yaml.parent / "labels" / args.split
        if not args.only:
            print(f"--limit {args.limit} : data.yaml temporaire {data_yaml}")

    speed_images = test_images[:args.n_speed_samples]

    if args.only:
        # Sous-process isolé pour un seul modèle : imprime le résultat en JSON et quitte
        # (la fin du process libère toute la VRAM, contrairement à del + empty_cache()).
        row = run_one(args.only, args, data_yaml, test_images, labels_dir, speed_images, device)
        print(f"{RESULT_MARKER}{json.dumps(row)}")
        return

    rows = []
    for name, tag in MODELS:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--only", tag,
               "--weights", args.weights, "--dataset", args.dataset, "--split", args.split,
               "--imgsz", str(args.imgsz), "--batch", str(args.batch), "--device", device,
               "--n-speed-samples", str(args.n_speed_samples)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            sys.exit(f"Échec de l'évaluation pour {name} (code {proc.returncode})")

        json_line = next(l for l in proc.stdout.splitlines() if l.startswith(RESULT_MARKER))
        row = json.loads(json_line[len(RESULT_MARKER):])
        row["name"] = name
        rows.append(row)

    # ── Tableau récapitulatif ─────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"{'Modèle':<18}{'Params(M)':>11}{'FLOPs(G)':>11}{'Inférence(ms)':>16}"
          f"{'mAP50':>10}{'mAP50-95':>11}{'P':>8}{'R':>8}")
    print("-" * 100)
    for r in rows:
        def fmt(v, pct=False):
            if v is None:
                return "N/A"
            return f"{v*100:.1f}%" if pct else f"{v:.2f}"
        print(f"{r['name']:<18}{fmt(r['params_m']):>11}{fmt(r['flops_g']):>11}"
              f"{fmt(r['speed_ms']):>16}{fmt(r['map50'], True):>10}"
              f"{fmt(r['map50_95'], True):>11}{fmt(r['precision'], True):>8}"
              f"{fmt(r['recall'], True):>8}")
    print("=" * 100)


if __name__ == "__main__":
    main()
