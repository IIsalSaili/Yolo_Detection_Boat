"""
Script d'entrainement YOLO-SEA optimisé pour la détection longue distance.

Deux générations de modèle :
    --gen 26 (défaut) : YOLO26-SEA — end-to-end NMS-free, SESA + BiFPN P2-P5 (petites cibles)
    --gen v8          : YOLOv8-SEA — SESA + BiFPN P3-P5 (architecture du papier Entropy 2025)

Usage :
    python train_sea.py                          # YOLO26-SEA, surveillance_marine, 300 epochs, imgsz=1280
    python train_sea.py --nwd                    # + blend NWD dans la loss/assigner (petites cibles)
    python train_sea.py --gen v8                 # architecture YOLOv8-SEA originale
    python train_sea.py --dataset fpp1           # dataset 21 classes (taxonomie projet complète)
    python train_sea.py --dataset 6classfinal    # dataset original 6 classes
    python train_sea.py --imgsz 640              # résolution réduite (plus rapide, moins précis)
    python train_sea.py --quick                  # test rapide 20 epochs
    python train_sea.py --baseline               # baseline (yolo26n ou yolov8n selon --gen)
    python train_sea.py --scale s                # scale 's'
    python train_sea.py --resume runs/sea/yolo_sea26_surveillance_n/weights/last.pt --nwd
                                                 # (re-passer --nwd au resume : flag non persisté)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent

DATASETS = {
    "surveillance": ROOT / "surveillance_marine" / "data.yaml",
    "6classfinal":  ROOT / "6classfinal" / "data.yaml",
    "fpp1":         ROOT / "fpp1" / "data.yaml",   # LabelDeZero, 21 classes (10 mil OTAN + 11 civ)
    # combined : fpp1 (jeu vidéo) + ship.v7 (réel), ~54k images. Toutes les boîtes sont
    # annotées en classe 20 (unknown) → s'entraîne en détection mono-classe avec --single-cls.
    "combined":     ROOT / "datasets" / "combined" / "data.yaml",
}

# Le scale est inséré dans le nom du yaml (ex. yolo26s-sea.yaml) : Ultralytics retire la lettre
# pour trouver le fichier de base (yolo26-sea.yaml) et applique les multiplicateurs du scale.
SEA_YAML = {
    "26": ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26{scale}-sea.yaml",
    "v8": ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8{scale}-sea.yaml",
}
SEA_YAML_BASE = {
    "26": ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-sea.yaml",
    "v8": ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-sea.yaml",
}
BASELINE_YAML = {"26": "yolo26{scale}.yaml", "v8": "yolov8{scale}.yaml"}

# Poids COCO pretrain pour initialiser le backbone (chargement partiel compatible)
PRETRAIN_PT = {"26": "yolo26{scale}.pt", "v8": "yolov8{scale}.pt"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", choices=["26", "v8"], default="26",
                   help="Génération du modèle : 26 = YOLO26-SEA NMS-free P2-P5 (défaut), "
                        "v8 = YOLOv8-SEA du papier")
    p.add_argument("--dataset", choices=["surveillance", "6classfinal", "fpp1", "combined"],
                   default="surveillance")
    p.add_argument("--single-cls", action="store_true",
                   help="Entrainer en détection mono-classe (toutes les boîtes → classe 0). "
                        "Requis pour 'combined' (annoté uniquement en classe unknown).")
    p.add_argument("--imgsz",    type=int, default=1280,
                   help="Résolution entrainement. 1280 recommandé pour longue distance (defaut), "
                        "640 si VRAM insuffisante")
    p.add_argument("--batch",    type=int, default=None,
                   help="Batch size (auto : gen 26 → 6 à 1280 [tête P2 plus lourde], "
                        "gen v8 → 8 à 1280 ; 16 à 640)")
    p.add_argument("--quick",    action="store_true", help="20 epochs pour tester le pipeline")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--scale",    choices=["n", "s"], default="n")
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--nwd",      action="store_true",
                   help="Blend NWD (Normalized Wasserstein Distance) dans la box loss et "
                        "l'assigner — recommandé pour les très petites cibles")
    p.add_argument("--nwd-ratio", type=float, default=0.5,
                   help="Part de NWD dans le blend (0 = CIoU pur, 1 = NWD pur ; défaut 0.5)")
    p.add_argument("--nwd-c",    type=float, default=12.8,
                   help="Constante C de la NWD en pixels (~taille moyenne des cibles ; défaut 12.8)")
    p.add_argument("--no-pretrain", action="store_true",
                   help="Ne pas charger les poids COCO pour le backbone")
    p.add_argument("--resume",   type=str, default=None)
    p.add_argument("--device",   type=str, default="auto",
                   help="'auto' (défaut : cuda si dispo sinon cpu), 'cpu', '0', '0,1' …")
    return p.parse_args()


def resolve_device(arg):
    """auto → 'cuda:0' si un GPU CUDA est dispo, sinon 'cpu'. Sinon valeur telle quelle."""
    if arg != "auto":
        return arg
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()

    data_yaml = DATASETS[args.dataset]
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml introuvable : {data_yaml}")
    if not args.baseline and not SEA_YAML_BASE[args.gen].exists():
        raise FileNotFoundError(f"Config sea introuvable : {SEA_YAML_BASE[args.gen]}")

    # Batch auto selon résolution et génération (RTX 4060 Ti 8 Go).
    # La tête P2 du YOLO26-SEA quadruple les anchors du niveau le plus fin → plus de VRAM.
    if args.batch is not None:
        batch = args.batch
    elif args.imgsz >= 1280:
        batch = 6 if args.gen == "26" else 8
    else:
        batch = 16

    if args.baseline:
        model_cfg = BASELINE_YAML[args.gen].format(scale=args.scale)
        run_name  = f"baseline_{'yolo26' if args.gen == '26' else 'yolov8'}{args.scale}_{args.dataset}"
    else:
        model_cfg = str(SEA_YAML[args.gen]).format(scale=args.scale)
        run_name  = f"yolo_sea{args.gen}_{args.dataset}_{args.scale}_i{args.imgsz}"
        if args.nwd:
            run_name += "_nwd"
        if args.single_cls:
            run_name += "_1cls"

    epochs = args.epochs if args.epochs is not None else (20 if args.quick else 300)

    device = resolve_device(args.device)
    if str(device) == "cpu":
        print("\n⚠️  Entrainement sur CPU : torch est un build CPU-only (ou --device cpu).")
        print("   À imgsz=1280/batch=6 c'est TRÈS lent. Pour tester : --quick --imgsz 640.\n")

    # NWD : flags globaux lus par BboxLoss (utils/loss.py) et TaskAlignedAssigner (utils/tal.py).
    if args.nwd:
        import ultralytics.utils.metrics as metrics
        metrics.USE_NWD = True
        metrics.NWD_RATIO = args.nwd_ratio
        metrics.NWD_C = args.nwd_c

    print(f"\n{'='*60}")
    print(f"Modele   : {model_cfg}")
    print(f"Dataset  : {data_yaml}")
    print(f"imgsz    : {args.imgsz}  batch={batch}")
    print(f"Epochs   : {epochs}")
    print(f"NWD      : {'ON (ratio=%.2f, C=%.1f)' % (args.nwd_ratio, args.nwd_c) if args.nwd else 'off'}")
    print(f"1-class  : {'ON (toutes les boîtes → classe 0)' if args.single_cls else 'off'}")
    print(f"Device   : {device}")
    print(f"Run      : {run_name}")
    print(f"{'='*60}\n")

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(model_cfg)
        # Charger les poids COCO pour le backbone (chargement partiel : seules les couches
        # qui correspondent par nom et par forme sont copiées — backbone layers 0-8).
        # SESA, BiFPN et la tête restent aléatoires.
        if not args.baseline and not args.no_pretrain:
            pretrain = PRETRAIN_PT[args.gen].format(scale=args.scale)
            print(f"Chargement backbone COCO ({pretrain}) …")
            model.load(pretrain)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        single_cls=args.single_cls,

        # ── Optimizer ────────────────────────────────────────────────
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,           # lr finale = lr0 * lrf (cosine schedule)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,    # warmup plus long : SESA+BiFPN partent de zéro
        warmup_momentum=0.8,
        cos_lr=True,        # cosine decay > step decay sur longue durée

        # ── Loss weights — priorité détection sur classification ──────
        # L'objectif est de détecter à longue distance ; la classification
        # viendra d'une tête dédiée plus tard.
        box=10.0,           # localisation prioritaire (défaut Ultralytics : 7.5)
        cls=0.3,            # classification moins pénalisée (défaut : 0.5)
        dfl=1.5,            # gen 26 (reg_max=1) : ce poids pondère la loss L1 de raffinement

        # ── Augmentations orientées petites cibles ───────────────────
        # copy_paste : colle des objets annotés dans d'autres images →
        # multiplie artificiellement les petits bateaux lointains.
        copy_paste=0.3,
        # scale plus large : le modèle voit des objets entre 10% et 90%
        # de la taille originale → apprend à détecter sur toute la gamme.
        scale=0.9,
        # Mosaic actif jusqu'aux 30 derniers epochs pour finir proprement.
        mosaic=1.0,
        close_mosaic=30,
        # Légère rotation maritime (tangage/roulis du bateau observateur).
        degrees=5.0,
        # Décalage de perspective et flip vertical désactivés : plan eau horizontal.
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        # Désaturation légère : brouillard, brume, contre-jour.
        hsv_s=0.5,
        hsv_v=0.4,
        # Random Erasing (masking) : efface un patch rectangulaire → robustesse
        # à l'occlusion + anti-overfit. NB : aveugle aux bbox, donc à 0.4 il peut
        # recouvrir entièrement une cible lointaine (quelques px) et créer un
        # label menteur. Valeur assumée ici ; baisser à 0.1 si les petites cibles
        # régressent.
        erasing=0.4,
        # MixUp léger : combinaison linéaire de 2 images+labels, en plus du mosaic
        # → régularisation supplémentaire pour quasi rien.
        mixup=0.1,

        # ── Eval / sauvegarde ─────────────────────────────────────────
        device=device,
        workers=8,
        project=str(ROOT / "runs" / "sea"),
        name=run_name,
        save=True,
        save_period=50,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n=== Entrainement termine ===")
    print(f"Meilleurs poids : {results.save_dir}/weights/best.pt")
    print(f"mAP@0.5        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"mAP@0.5:0.95   : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
