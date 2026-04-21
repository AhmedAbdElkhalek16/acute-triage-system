"""
app.py
------
Gradio demo UI for the Acute Findings Triage System.

Run locally:
    python app.py

Then open http://localhost:7860 in your browser.
Upload an X-Ray or CT image and get:
  - Priority level (CRITICAL / HIGH / MEDIUM / LOW)
  - Confidence score per condition
  - Grad-CAM heatmap showing where the model focused
"""

import gradio as gr
import torch
import numpy as np
import cv2
from pathlib import Path

from src.preprocessing import get_val_transforms, load_image
from src.models import get_model
from src.triage_engine import TriageEngine
from src.gradcam import get_gradcam, TARGET_LAYERS


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
XRAY_WEIGHTS   = 'models/weights/xray_best.pth'
CT_WEIGHTS     = 'models/weights/ct_best.pth'
IMAGE_SIZE     = 512

PRIORITY_COLORS = {
    'CRITICAL': '#E24B4A',
    'HIGH'    : '#EF9F27',
    'MEDIUM'  : '#378ADD',
    'LOW'     : '#1D9E75',
}

# ─────────────────────────────────────────────
#  Load Models (cached at startup)
# ─────────────────────────────────────────────

def load_models():
    models = {}
    transforms = get_val_transforms(IMAGE_SIZE)

    for modality, weights_path, model_name in [
        ('xray', XRAY_WEIGHTS, 'efficientnet_b4'),
        ('ct',   CT_WEIGHTS,   'densenet121'),
    ]:
        model = get_model(modality, num_classes=3, pretrained=False,
                          device=DEVICE)
        if Path(weights_path).exists():
            model.load(weights_path, device=DEVICE)
            print(f"[app] Loaded {modality} weights from {weights_path}")
        else:
            print(f"[app] WARNING: {weights_path} not found — using random weights for demo")

        model.eval()
        models[modality] = {
            'model'     : model,
            'transforms': transforms,
            'name'      : model_name,
        }

    return models


MODELS  = load_models()
ENGINE  = TriageEngine()


# ─────────────────────────────────────────────
#  Core Inference Function
# ─────────────────────────────────────────────

def run_triage(image: np.ndarray, modality: str) -> tuple:
    """
    Run triage pipeline on an uploaded image.

    Args:
        image    : numpy RGB array from Gradio
        modality : 'xray' or 'ct'

    Returns:
        tuple of (heatmap_image, report_html, priority_badge)
    """
    if image is None:
        return None, "<p>No image uploaded.</p>", ""

    m          = MODELS[modality]
    model      = m['model']
    transforms = m['transforms']
    model_name = m['name']

    # 1. Preprocess
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    tensor  = transforms(image=image)['image'].unsqueeze(0).to(DEVICE)

    # 2. Get probabilities
    probs = model.predict_proba(tensor)

    # 3. Grad-CAM
    cam = get_gradcam(model, model_name)
    _, overlay, _ = cam.generate(tensor, class_idx=probs.argmax().item(),
                                 original_image=img_bgr)
    cam.remove_hooks()
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 4. Triage report
    report = ENGINE.evaluate(probs, modality=modality)

    # 5. Build HTML report
    html  = _build_report_html(report)
    badge = report.overall_priority.label

    return overlay_rgb, html, badge


# ─────────────────────────────────────────────
#  HTML Report Builder
# ─────────────────────────────────────────────

def _build_report_html(report) -> str:
    priority = report.overall_priority
    color    = PRIORITY_COLORS.get(priority.label, '#888')

    rows = ""
    if report.findings:
        for f in report.findings:
            fc = PRIORITY_COLORS.get(f.priority.label, '#888')
            rows += f"""
            <tr>
              <td style='padding:8px 12px; font-weight:500'>{f.condition.name}</td>
              <td style='padding:8px 12px'>
                <span style='background:{fc}22; color:{fc}; padding:3px 10px;
                             border-radius:12px; font-size:12px; font-weight:500'>
                  {f.priority.label}
                </span>
              </td>
              <td style='padding:8px 12px'>{f.confidence*100:.1f}%</td>
              <td style='padding:8px 12px; color:#888; font-size:12px'>{f.condition.description}</td>
            </tr>"""
    else:
        rows = "<tr><td colspan='4' style='padding:12px; color:#888'>No acute findings detected.</td></tr>"

    return f"""
    <div style='font-family:sans-serif; padding:8px'>
      <div style='background:{color}15; border-left:4px solid {color};
                  padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:16px'>
        <div style='font-size:11px; color:{color}; font-weight:500; text-transform:uppercase;
                    letter-spacing:1px'>Overall Priority</div>
        <div style='font-size:24px; font-weight:500; color:{color}'>{priority.label}</div>
        <div style='font-size:13px; color:#666; margin-top:4px'>
          Response required: {priority.response_time}
        </div>
      </div>
      <table style='width:100%; border-collapse:collapse; font-size:14px'>
        <thead>
          <tr style='background:#f5f5f5; font-size:11px; color:#888; text-transform:uppercase'>
            <th style='padding:8px 12px; text-align:left'>Condition</th>
            <th style='padding:8px 12px; text-align:left'>Priority</th>
            <th style='padding:8px 12px; text-align:left'>Confidence</th>
            <th style='padding:8px 12px; text-align:left'>Notes</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style='font-size:11px; color:#aaa; margin-top:16px'>
        ⚕ For research and educational purposes only.
        Not for clinical use.
      </p>
    </div>"""


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="Acute Findings Triage System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("## Acute Findings Triage System")
    gr.Markdown("Upload a chest X-Ray or CT scan. The system detects critical findings and assigns a triage priority.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Medical Image (PNG / DICOM)",
                type="numpy",
                height=400,
            )
            modality_radio = gr.Radio(
                choices=["xray", "ct"],
                value="xray",
                label="Image modality",
            )
            run_btn = gr.Button("Run Triage", variant="primary")

        with gr.Column(scale=1):
            heatmap_output = gr.Image(
                label="Grad-CAM — model focus area",
                height=400,
            )

    report_output = gr.HTML(label="Triage Report")

    run_btn.click(
        fn=run_triage,
        inputs=[image_input, modality_radio],
        outputs=[heatmap_output, report_output, gr.Textbox(visible=False)],
    )

    gr.Examples(
        examples=[],
        inputs=image_input,
    )


if __name__ == '__main__':
    demo.launch(share=False, server_port=7860)