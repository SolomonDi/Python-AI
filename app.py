from pathlib import Path

import gradio as gr

from inference import Predictor

predictor = Predictor()

SAMPLES_DIR = Path(__file__).with_name("samples")
EXAMPLES = sorted(SAMPLES_DIR.glob("normal_*.jpeg")) + sorted(SAMPLES_DIR.glob("pneumonia_*.jpeg"))

CSS = """
.verdict {padding: 18px 20px; border-radius: 12px; font-size: 1.05rem;}
.verdict h2 {margin: 0 0 6px 0;}
.verdict.normal {background: rgba(34, 160, 90, 0.12); border: 1px solid rgba(34, 160, 90, 0.5);}
.verdict.pneumonia {background: rgba(220, 60, 60, 0.12); border: 1px solid rgba(220, 60, 60, 0.5);}
"""


def verdict_html(prob, threshold):
    if prob is None:
        return "<div class='verdict'>Upload a chest X-ray and press <b>Analyze</b>.</div>"
    if prob > threshold:
        cls, title = "pneumonia", "Pneumonia"
    else:
        cls, title = "normal", "Normal"
    return (f"<div class='verdict {cls}'><h2>{title}</h2>"
            f"Pneumonia probability: <b>{prob * 100:.1f}%</b> · threshold {threshold * 100:.0f}%</div>")


def analyze(image, threshold):
    if image is None:
        return None, verdict_html(None, threshold), None, None
    prob, cam = predictor.predict_with_cam(image)
    label = {"Pneumonia": prob, "Normal": 1 - prob}
    return prob, verdict_html(prob, threshold), label, cam


with gr.Blocks(title="PneumoScan") as demo:
    gr.Markdown("# PneumoScan\n"
                "AI-assisted pneumonia screening for chest radiographs. "
                "The Grad-CAM heatmap highlights the regions that drove the prediction.")

    prob_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Chest X-ray", height=420)
            threshold = gr.Slider(0.01, 0.99, value=round(predictor.threshold, 2), step=0.01,
                                  label="Decision threshold",
                                  info="Lower: higher sensitivity · Higher: fewer false positives")
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                run_btn = gr.Button("Analyze", variant="primary")
            gr.Examples(examples=[str(p) for p in EXAMPLES], inputs=image_in,
                        label="Sample studies (first 3 normal, last 3 pneumonia)")

        with gr.Column():
            verdict = gr.HTML(verdict_html(None, predictor.threshold))
            probs = gr.Label(label="Probabilities", num_top_classes=2)
            cam_out = gr.Image(label="Heatmap (Grad-CAM)", height=420, interactive=False)

    gr.Markdown("<small>For research use only. Not intended for clinical diagnosis.</small>")

    outputs = [prob_state, verdict, probs, cam_out]
    run_btn.click(analyze, [image_in, threshold], outputs)
    image_in.upload(analyze, [image_in, threshold], outputs)
    threshold.change(verdict_html, [prob_state, threshold], verdict)
    clear_btn.add([image_in, cam_out, probs])
    clear_btn.click(lambda t: (None, verdict_html(None, t)), threshold, [prob_state, verdict])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
