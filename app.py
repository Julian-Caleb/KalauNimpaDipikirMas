import gradio as gr
import torch
import torch.nn.functional as F
import easyocr
from PIL import Image
from transformers import BertTokenizer

from config import DEVICE, BERT_MODEL_NAME, MAX_SEQ_LEN
from model import MultimodalFakeNewsDetector
from preprocessing import preprocess_ocr_text, eval_transform

# List of models
MODEL_REGISTRY = [
    ("Multimodal",   "model/Multimodal.pt",   "multimodal",  "Fake", "Real"),
    ("Unimodal Visual",   "model/Vision.pt",   "vision_only",  "Fake", "Real"),
    ("Unimodal Textual",   "model/Text.pt",   "text_only",  "Fake", "Real"),
]

_PLACEHOLDER = "(Select model)"
DROPDOWN_CHOICES = [_PLACEHOLDER] + [entry[0] for entry in MODEL_REGISTRY]
_REGISTRY_MAP = {entry[0]: entry for entry in MODEL_REGISTRY}

# Application state
state = {
    "model":      None,
    "tokenizer":  None,
    "ocr_reader": None,
    "mode":       None,
    "label_map":  None,
}


def load_model(model_name):
    if model_name == _PLACEHOLDER or model_name is None:
        return "⚠️ Select a model first."

    _, model_path, mode, label_fake, label_real = _REGISTRY_MAP[model_name]

    try:
        tokenizer  = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        ocr_reader = easyocr.Reader(["id", "en"], gpu=torch.cuda.is_available())

        model = MultimodalFakeNewsDetector(mode=mode).to(DEVICE)

        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        state.update(
            model=model,
            tokenizer=tokenizer,
            ocr_reader=ocr_reader,
            mode=mode,
            label_map={0: label_real, 1: label_fake},
        )
        return f"✅ Model successfully loaded\nName  : {model_name}\nMode  : {mode}"

    except FileNotFoundError:
        state.update(model=None, tokenizer=None, ocr_reader=None, mode=None, label_map=None)
        return f"❌ File not found:\n{model_path}"
    except RuntimeError as e:
        state.update(model=None, tokenizer=None, ocr_reader=None, mode=None, label_map=None)
        return f"❌ Failed to load model weights (possible architecture mismatch):\n{e}"
    except Exception as e:
        state.update(model=None, tokenizer=None, ocr_reader=None, mode=None, label_map=None)
        return f"❌ Failed to load model:\n{e}"


def classify(image):
    if state["model"] is None:
        return (
            "⚠️ Model not loaded.",
            {"Error": 1.0},
            ""
        )

    if image is None:
        return (
            "⚠️ Please upload an image first.",
            {"Error": 1.0},
            ""
        )

    try:
        img_t = eval_transform(
            Image.fromarray(image).convert("RGB")
        ).unsqueeze(0).to(DEVICE)

        raw_text = " ".join(
            state["ocr_reader"].readtext(
                image,
                detail=0,
                paragraph=True
            )
        )

        text = preprocess_ocr_text(raw_text)

        enc = state["tokenizer"](
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = state["model"](
                img_t,
                input_ids,
                attention_mask
            )

            probs = F.softmax(logits, dim=-1)[0]
            pred = logits.argmax(dim=-1).item()

            pred_label = state["label_map"][pred]

            confidence = {
                state["label_map"][0]: float(probs[0]),
                state["label_map"][1]: float(probs[1]),
            }

            show_ocr = state["mode"] in [
                "multimodal",
                "text_only"
            ]

            if show_ocr:
                ocr_text = text if text else "(No text detected)"
            else:
                ocr_text = f"(Mode '{state['mode']}' does not use OCR)"

            return (
                pred_label,
                confidence,
                ocr_text
            )

    except Exception as e:
        return (
            f"❌ Error: {e}",
            {"Error": 1.0},
            ""
        )


# Gradio UI
css = """
.container {
    max-width: 1200px;
    margin: auto;
}

.section-title {
    padding-left: 10px;
}

.section-title h3 {
    margin-top: 6px;
    margin-bottom: 6px;
}

@media (max-width: 768px) {
    .section-title {
        padding-left: 8px;
    }

    .section-title h3 {
        font-size: 1rem;
    }
}
"""

with gr.Blocks(
    title="TEXT OVERWRITING MANIPULATION DETECTOR",
    css=css,
    theme=gr.themes.Default(),
) as demo:

    gr.Markdown(
        """
        # TEXT OVERWRITING MANIPULATION DETECTOR
        """
    )

    with gr.Row(equal_height=True):

        # Load Image & Classify Button
        with gr.Column(scale=1, min_width=350):

            image_input = gr.Image(
                label="Image",
                height=320,
                type="numpy"
            )

            btn_classify = gr.Button(
                "CLASSIFY",
                variant="primary"
            )

        # Model Info & Classification Result
        with gr.Column(scale=1, min_width=350):

            # Load Model
            with gr.Group():

                gr.HTML("""
                <div class="section-title">
                    <h3>MODEL USED</h3>
                </div>
                """)

                model_select = gr.Dropdown(
                    choices=DROPDOWN_CHOICES,
                    value=_PLACEHOLDER,
                    label="Select Model"
                )

                btn_load = gr.Button(
                    "LOAD MODEL", 
                    variant="primary"
                )

                model_info = gr.Textbox(
                    label=None,
                    show_label=False,
                    interactive=False,
                    lines=3
                )

            # Classification Result
            with gr.Group():

                gr.HTML("""
                <div class="section-title">
                    <h3>CLASSIFICATION RESULTS</h3>
                </div>
                """)

                prediction_box = gr.Textbox(
                    label="Prediction",
                    interactive=False
                )

                confidence_box = gr.Label(
                    label="Confidence Score",
                    num_top_classes=2,
                    container=False
                )

                ocr_box = gr.Textbox(
                    label="OCR Results",
                    interactive=False,
                    lines=4
                )

    # Events
    btn_load.click(
        fn=load_model,
        inputs=model_select,
        outputs=model_info
    )

    btn_classify.click(
        fn=classify,
        inputs=image_input,
        outputs=[
            prediction_box,
            confidence_box,
            ocr_box
        ]
    )

if __name__ == "__main__":
    demo.launch()