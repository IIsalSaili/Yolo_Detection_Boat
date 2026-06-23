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
BASELINE_WEIGHTS = "yolo26s.pt"  # stock, pré-entraîné COCO, téléchargé auto par ultralytics

# Chargés une seule fois (pas à chaque image).
model = YOLO(str(WEIGHTS))
baseline_model = YOLO(BASELINE_WEIGHTS)


def _run(model, image, conf, iou, imgsz):
    """Inférence sur une image PIL → image annotée + texte récap."""
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=int(imgsz),
        verbose=False,
    )
    r = results[0]

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


def detect(image, conf, iou, imgsz):
    """Inférence comparée : modèle custom (best.pt) vs baseline YOLO26s (COCO)."""
    if image is None:
        return None, "Charge une image d'abord.", None, "Charge une image d'abord."

    custom_img, custom_summary = _run(model, image, conf, iou, imgsz)
    baseline_img, baseline_summary = _run(baseline_model, image, conf, iou, imgsz)

    return custom_img, custom_summary, baseline_img, baseline_summary


with gr.Blocks(title="YOLO-SEA — Détection navires") as demo:
    gr.Markdown(
        "# 🚢 YOLO-SEA — Détection de navires\n"
        "Charge une image (RGB ou IR) : comparaison entre le modèle custom et une baseline YOLO26 standard.\n"
        f"*Custom : `{WEIGHTS.name}` (mono-classe, fine-tuné navires) — Baseline : `{BASELINE_WEIGHTS}` (COCO, non fine-tuné).*"
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
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Modèle custom (YOLO-SEA)")
            out_img = gr.Image(label="Détections — custom")
            out_txt = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Baseline YOLO26s (COCO)")
            out_img_baseline = gr.Image(label="Détections — baseline")
            out_txt_baseline = gr.Markdown()

    outputs = [out_img, out_txt, out_img_baseline, out_txt_baseline]
    btn.click(detect, [inp, conf, iou, imgsz], outputs)
    inp.upload(detect, [inp, conf, iou, imgsz], outputs)


if __name__ == "__main__":
    demo.launch()
