"""
app.py
======
Loads the saved MARBERTv2 weights (marbert_sentiment.pt)
and launches a Gradio interface — no training required.

Prerequisites:
    pip install -r requirements.txt
"""

import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
PT_PATH        = "marbert_sentiment.pt"
TOKENIZER_PATH = "marbert_tokenizer"      # folder saved by train_and_save.py
MAX_LENGTH     = 128
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Load tokenizer & model ────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print("Loading model weights...")
model = AutoModelForSequenceClassification.from_pretrained(
    "UBC-NLP/MARBERTv2",   # architecture only — weights come from .pt
    num_labels=2,
    problem_type="single_label_classification",
)
model.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model ready!\n")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "❌ الرجاء إدخال نص"

    cleaned = preprocess(text)
    inputs  = tokenizer(
        cleaned,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=1)[0]
    pred_label = probs.argmax().item()        # 0 = negative, 1 = positive
    confidence = probs[pred_label].item()

    if pred_label == 1:
        return f"✅ 😊 **إيجابي**\n\nالثقة: **{confidence*100:.1f}%**"
    else:
        return f"❌ 😞 **سلبي**\n\nالثقة: **{confidence*100:.1f}%**"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
iface = gr.Interface(
    fn=predict_sentiment,
    title="   تحليل المشاعر ",
    description="اكتب أي تعليق بالعامية المصرية وهيقولك إيجابي ولا سلبي",
    inputs=gr.Textbox(
        lines=5,
        placeholder="مثال: التطبيق ده تحفة والله، التوصيل سريع جدا",
        label="النص",
    ),
    outputs=gr.Markdown(label="النتيجة"),
    examples=[
        ["التطبيق ده تحفة والله، التوصيل سريع جدا"],
        ["خدمة سيئة جداً، الطلب اتأخر 3 ساعات"],
        ["والله العظيم أحسن تطبيق جربتو"],
        ["بيغلط كتير ومش بيردوا على الشكاوى"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=True)