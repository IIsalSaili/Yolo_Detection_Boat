"""
Interface d'inférence YOLO-SEA — détection de navires longue distance.

Charge le checkpoint entraîné (best.pt) et détecte les cibles sur une image.
Utilise le `ultralytics` LOCAL du repo (modules SESA / BiFPN custom).

Lancer :
    .venv/bin/python app.py
    # puis ouvrir l'URL affichée (http://127.0.0.1:7860)
"""

from pathlib import Path

import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights" / "best.pt"

# Chargé une seule fois (pas à chaque image).
model = YOLO(str(WEIGHTS))


def detect(image, conf, iou, imgsz):
    """Inférence sur une image PIL → image annotée + texte récap."""
    if image is None:
        return None, "Charge une image d'abord."

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        verbose=False,
    )
    r = results[0]

    # Image annotée (boîtes + scores) en RGB pour gradio.
    annotated = r.plot()[:, :, ::-1]  # BGR (OpenCV) -> RGB

    n = len(r.boxes)
    if n == 0:
        summary = "Aucune détection (baisse le seuil de confiance ?)."
    else:
        confs = r.boxes.conf.tolist()
        lines = [f"**{n} détection(s)**", ""]
        for i, c in enumerate(sorted(confs, reverse=True), 1):
            lines.append(f"{i}. confiance = {c:.2%}")
        summary = "\n".join(lines)

    return annotated, summary


with gr.Blocks(title="YOLO-SEA — Détection navires") as demo:
    gr.Markdown(
        "# 🚢 YOLO-SEA — Détection de navires\n"
        "Charge une image (RGB ou IR), le modèle entoure les cibles détectées.\n"
        f"*Poids : `{WEIGHTS.name}` — mono-classe.*"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Image d'entrée")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                             label="Seuil de confiance")
            iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                            label="Seuil IoU (NMS)")
            imgsz = gr.Radio([640, 960, 1280], value=640,
                             label="Résolution d'inférence (640 = entraînement)")
            btn = gr.Button("Détecter", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Détections")
            out_txt = gr.Markdown()

    btn.click(detect, [inp, conf, iou, imgsz], [out_img, out_txt])
    inp.upload(detect, [inp, conf, iou, imgsz], [out_img, out_txt])


if __name__ == "__main__":
    demo.launch()
