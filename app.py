"""
app.py - the demo screen: place two images side by side and check which one is AI-generated.

    python app.py            open the demo at http://127.0.0.1:7860
    python app.py --share    also create a public link (use this on Colab)
"""
import argparse
import json
from pathlib import Path

import gradio as gr

from detector import analyze, load_detector

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "detector.pt"
if not MODEL_FILE.exists():
    raise SystemExit("models/detector.pt not found. Train it first:  python train.py train")
MODEL = load_detector(MODEL_FILE, device="cpu")  # a light model: the CPU is fast enough for a live demo

UNCERTAIN_BAND = 0.15  # probabilities within 50% +/- this are reported as "uncertain"

CSS = """
.verdict {border-radius: 14px; padding: 16px 18px; font-family: system-ui, sans-serif;}
.verdict, .verdict * {color: #fff !important;}
.verdict.real {background: #17804d;} .verdict.fake {background: #c0392b;} .verdict.unsure {background: #b7791f;}
.verdict .tag {font-size: 26px; font-weight: 800; letter-spacing: .5px;}
.verdict .pct {font-size: 15px; margin-top: 2px; opacity: .95;}
.verdict .bar {height: 8px; border-radius: 4px; background: rgba(255,255,255,.3); margin: 10px 0 8px;}
.verdict .bar div {height: 100%; border-radius: 4px; background: #fff;}
.verdict .meta {font-size: 14px; opacity: .95;}
.summary {text-align: center; font-size: 20px; font-weight: 700; padding: 12px; border-radius: 12px;
          border: 2px solid var(--border-color-primary);}
.empty {text-align: center; opacity: .6; padding: 22px;}
"""


def verdict_card(result):
    """HTML card for one analysed image."""
    if result is None:
        return '<div class="empty">No image</div>'
    p = result["p_fake"]
    is_fake = p >= 0.5
    if abs(p - 0.5) < UNCERTAIN_BAND:
        style, tag = "unsure", f"UNCERTAIN (leaning {'AI' if is_fake else 'real'})"
    else:
        style, tag = ("fake", "AI-GENERATED") if is_fake else ("real", "REAL PHOTO")
    meta = "No generator fingerprint found."
    if is_fake:
        generator, share = max(result["sources"].items(), key=lambda kv: kv[1])
        meta = (f"Looks most like <b>{generator}</b> ({share:.0%} match)" if share >= 0.5 else
                f"No clear generator fingerprint (closest: {generator}, {share:.0%})")
    return (f'<div class="verdict {style}"><div class="tag">{tag}</div>'
            f'<div class="pct">Probability of being AI-generated: <b>{p:.1%}</b></div>'
            f'<div class="bar"><div style="width:{p * 100:.1f}%"></div></div>'
            f'<div class="meta">{meta}</div></div>')


def summary_line(a, b):
    """One sentence comparing the two images."""
    if a is None or b is None:
        return ""
    fake_a, fake_b = a["p_fake"] >= 0.5, b["p_fake"] >= 0.5
    if fake_a and fake_b:
        text = "Both images look AI-generated."
    elif not fake_a and not fake_b:
        text = "Both images look like real photos."
    else:
        text = f"Image {'A' if fake_a else 'B'} is the AI-generated one; image {'B' if fake_a else 'A'} is the real photo."
    return f'<div class="summary">{text}</div>'


def check(image_a, image_b):
    if image_a is None and image_b is None:
        raise gr.Error("Add at least one image first.")
    a, b = [analyze(MODEL, image) if image is not None else None for image in (image_a, image_b)]
    return (summary_line(a, b), verdict_card(a), a and a["heatmap"], verdict_card(b), b and b["heatmap"])


def example_pairs():
    """examples/pairN_real.jpg + examples/pairN_<generator>.jpg, alternating which side is real."""
    pairs = []
    for k, real in enumerate(sorted((ROOT / "examples").glob("pair*_real.jpg"))):
        fake = next(p for p in real.parent.glob(real.name.split("_")[0] + "_*.jpg") if p != real)
        pairs.append([str(real), str(fake)] if k % 2 == 0 else [str(fake), str(real)])
    return pairs


def model_facts():
    """One line of test-set facts for the header, read from the last evaluation."""
    metrics_file = ROOT / "results" / "detector" / "metrics.json"
    if not metrics_file.exists():
        return ""
    m = json.loads(metrics_file.read_text())
    backbone = m["backbone"].replace("mobilenet_v", "MobileNetV").replace("efficientnet_b", "EfficientNet-B") \
        .replace("_large", "-Large").replace("_small", "-Small")
    return (f"Model: {backbone} ({m['cost']['params_millions']:.1f} M parameters, "
            f"{m['cost']['cpu_ms_per_image']:.0f} ms per image on CPU) &nbsp;|&nbsp; "
            f"test accuracy {m['accuracy']:.1%} on {m['test_images']} unseen images")


def build():
    with gr.Blocks(title="Real or AI? Deepfake image detector") as demo:
        gr.Markdown("# Real or AI?\nPlace two images side by side. The detector says which one is a real photo "
                    "and which one was AI-generated, names the generator it resembles, and shows where it looked.\n\n"
                    + model_facts())
        with gr.Row():
            image_a = gr.Image(type="pil", label="Image A", sources=["upload", "clipboard"], height=340)
            image_b = gr.Image(type="pil", label="Image B", sources=["upload", "clipboard"], height=340)
        button = gr.Button("Check both images", variant="primary", size="lg")
        summary = gr.HTML()
        with gr.Row():
            with gr.Column():
                card_a = gr.HTML()
                heat_a = gr.Image(label="Heat-map A: where the model looked", interactive=False, height=340)
            with gr.Column():
                card_b = gr.HTML()
                heat_b = gr.Image(label="Heat-map B: where the model looked", interactive=False, height=340)
        outputs = [summary, card_a, heat_a, card_b, heat_b]
        button.click(check, [image_a, image_b], outputs)
        pairs = example_pairs()
        if pairs:
            gr.Examples(pairs, [image_a, image_b], outputs, fn=check, run_on_click=True, cache_examples=False,
                        label="Example pairs from the unseen test data: the same scene as a real photo and as an AI image")
        gr.Markdown("Heat-map colours: red = regions that weighed most in the verdict, blue = ignored.  \n"
                    "Trained on real MS COCO photos and images from Stable Diffusion 2.1, SDXL, Stable Diffusion 3, "
                    "DALL-E 3 and Midjourney v6. Images from other generators, or heavily edited ones, can fool it.")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public link (needed on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab automatically")
    args = parser.parse_args()
    build().launch(share=args.share, server_port=args.port, inbrowser=not (args.share or args.no_browser),
                   css=CSS, theme=gr.themes.Soft())
