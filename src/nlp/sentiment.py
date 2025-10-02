from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_MODEL = "yiyanghkust/finbert-tone"
_tok = AutoTokenizer.from_pretrained(_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL)
_model.eval()

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def score_texts(texts):
    if not texts:
        return []
    enc = _tok(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels = [LABELS[i] for i in probs.argmax(axis=1)]
    scores = probs.max(axis=1)  # confidence of predicted label
    return list(zip(labels, scores))
