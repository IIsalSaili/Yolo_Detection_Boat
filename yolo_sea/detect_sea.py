"""
Visualisation des détections YOLO-SEA avec boîtes colorées par classe.

Usage :
    python detect_sea.py image.jpg
    python detect_sea.py image.jpg --dataset 6classfinal
    python detect_sea.py image.jpg --weights runs/sea/yolo_sea_surveillance_n/weights/best.pt
    python detect_sea.py video.mp4 --conf 0.3
    python detect_sea.py 0            # webcam
    python detect_sea.py image.jpg --soft-nms
    python detect_sea.py image.jpg --save result.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATASET_CFG = {
    "surveillance": {
        "classes": ["boat", "buoy", "kayak", "sail-boat", "speed-boat", "vessel", "war-ship"],
        "colors": {
            "boat":       (0,   200, 255),
            "buoy":       (0,   140, 255),
            "kayak":      (255, 255,   0),
            "sail-boat":  (0,   255, 180),
            "speed-boat": (0,    80, 255),
            "vessel":     (255,   0, 200),
            "war-ship":   (0,     0, 255),
        },
        "default_weights": Path("runs/sea/yolo_sea_surveillance_n/weights/yolo26m_chine.pt"),
    },
    "6classfinal": {
        "classes": ["a", "buoy", "container", "cruise", "fish-b", "warship"],
        "colors": {
            "a":         (0,   255, 255),
            "buoy":      (0,   165, 255),
            "container": (0,   255,   0),
            "cruise":    (255,   0, 255),
            "fish-b":    (255, 255,   0),
            "warship":   (0,     0, 255),
        },
        "default_weights": Path("runs/sea/yolo_sea_surveillance_n/weights/yolo26m_chine.pt"),
    },
}


def draw_boxes(frame: np.ndarray, results, classes: list, colors: dict) -> np.ndarray:
    out = frame.copy()
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return out

    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        color  = colors.get(label, (200, 200, 200))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def process_image(model, path: str, conf: float, save: str | None, classes: list, colors: dict):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Impossible de lire l'image : {path}")

    results = model.predict(frame, conf=conf, verbose=False)
    out = draw_boxes(frame, results, classes, colors)

    n = len(results[0].boxes) if results[0].boxes else 0
    print(f"{n} détection(s) — {Path(path).name}")

    if save:
        cv2.imwrite(save, out)
        print(f"Sauvegardé : {save}")

    cv2.imshow("YOLO-SEA — détections", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model, source: str, conf: float, save: str | None, classes: list, colors: dict):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la source : {source}")

    writer = None
    if save:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save, fourcc, fps, (w, h))

    print("Lecture en cours — appuyer sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        out = draw_boxes(frame, results, classes, colors)

        if writer:
            writer.write(out)

        cv2.imshow("YOLO-SEA — détections", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Vidéo sauvegardée : {save}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Image, vidéo ou index webcam (0, 1, ...)")
    p.add_argument("--dataset", choices=["surveillance", "6classfinal"],
                   default="surveillance",
                   help="Dataset utilisé pour les noms de classes (defaut : surveillance)")
    p.add_argument("--weights", default=None,
                   help="Poids du modèle (defaut : déduit du --dataset)")
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--soft-nms", action="store_true",
                   help="Activer Soft-NMS gaussien")
    p.add_argument("--save",     default=None, help="Chemin de sortie (image ou vidéo)")
    args = p.parse_args()

    cfg     = DATASET_CFG[args.dataset]
    classes = cfg["classes"]
    colors  = cfg["colors"]
    weights = Path(args.weights) if args.weights else cfg["default_weights"]

    if not weights.exists():
        sys.exit(
            f"Poids introuvables : {weights}\n"
            f"Lancer d'abord : python train_sea.py --dataset {args.dataset}"
        )

    model = YOLO(str(weights))
    print(f"Dataset  : {args.dataset}  ({len(classes)} classes : {', '.join(classes)})")
    print(f"Poids    : {weights}")

    if args.soft_nms:
        # Les modèles YOLO26 (end2end) n'utilisent pas de NMS : le flag est sans effet.
        if getattr(model.model, "end2end", False):
            print("Modèle end-to-end (NMS-free) : --soft-nms ignoré.")
        else:
            import ultralytics.utils.nms as nms_module
            nms_module.USE_SOFT_NMS = True
            print("Soft-NMS activé (gaussien).")

    source   = args.source
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if Path(source).suffix.lower() in img_exts:
        process_image(model, source, args.conf, args.save, classes, colors)
    else:
        process_video(model, source, args.conf, args.save, classes, colors)


if __name__ == "__main__":
    main()
