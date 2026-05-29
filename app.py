import gradio as gr
import torch
import torch.nn.functional as F
import easyocr
from PIL import Image
from transformers import BertTokenizer

from config import DEVICE, BERT_MODEL, MAX_SEQ_LEN
from model import MultimodalFakeNewsDetector
from preprocessing import preprocess_ocr_text, eval_transform

state = {"model": None, "tokenizer": None, "ocr_reader": None, "mode": None, "label_map": None}


def load_model(model_path, mode, label_fake, label_real):
    try:
        tokenizer  = BertTokenizer.from_pretrained(BERT_MODEL)
        ocr_reader = easyocr.Reader(["id", "en"], gpu=torch.cuda.is_available())
        model      = MultimodalFakeNewsDetector(mode=mode).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        state.update(model=model, tokenizer=tokenizer, ocr_reader=ocr_reader,
                     mode=mode, label_map={0: label_fake, 1: label_real})
        return f"✅ Model berhasil dimuat\nMode: {mode} | Device: {DEVICE}"
    except Exception as e:
        return f"❌ Gagal memuat model:\n{e}"


def classify(image):
    if state["model"] is None:
        return "⚠️ Model belum dimuat. Muat model terlebih dahulu."
    if image is None:
        return "⚠️ Unggah citra terlebih dahulu."
    try:
        img_t  = eval_transform(Image.fromarray(image).convert("RGB")).unsqueeze(0).to(DEVICE)
        text   = preprocess_ocr_text(
            " ".join(state["ocr_reader"].readtext(image, detail=0, paragraph=True)))
        enc    = state["tokenizer"](text, max_length=MAX_SEQ_LEN, padding="max_length",
                                    truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = state["model"](img_t, input_ids, attention_mask)
            probs  = F.softmax(logits, dim=-1)[0]
            pred   = logits.argmax(dim=-1).item()

        lf, lr = state["label_map"][0], state["label_map"][1]
        return (f"Prediksi  : {state['label_map'][pred]}\n"
                f"{lf:<12}: {probs[0].item()*100:.1f}%\n"
                f"{lr:<12}: {probs[1].item()*100:.1f}%\n"
                f"Teks OCR  : {text if text else '(tidak terdeteksi)'}")
    except Exception as e:
        return f"❌ Error saat klasifikasi:\n{e}"


with gr.Blocks(title="Deteksi Citra Berita") as demo:
    gr.Markdown("## DETEKSI CITRA BERITA MENGANDUNG TEKS")

    with gr.Row():
        with gr.Column(scale=1):
            image_input  = gr.Image(label="Citra", height=320)
            btn_classify = gr.Button("KLASIFIKASI", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**MODEL YANG DIGUNAKAN**")
                model_path  = gr.Textbox(label="Path Model (.pt)", placeholder="model/vision_only.pt")
                mode_select = gr.Radio(["multimodal", "vision_only", "text_only"],
                                       label="Mode", value="multimodal")
                with gr.Row():
                    label_fake = gr.Textbox(label="Label Kelas 0 (Fake)", value="Fake")
                    label_real = gr.Textbox(label="Label Kelas 1 (Real)", value="Real")
                btn_load   = gr.Button("MUAT MODEL")
                model_info = gr.Textbox(label="Status Model", interactive=False)

            with gr.Group():
                gr.Markdown("**HASIL KLASIFIKASI**")
                result_box = gr.Textbox(label="", interactive=False, lines=5)

    btn_load.click(load_model, inputs=[model_path, mode_select, label_fake, label_real],
                   outputs=model_info)
    btn_classify.click(classify, inputs=image_input, outputs=result_box)

if __name__ == "__main__":
    demo.launch()