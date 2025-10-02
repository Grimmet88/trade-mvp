from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_MODEL = "yiyanghkust/finbert-tone"

_tok = AutoTokenizer.from_pretrained(_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL)
_model.eval()

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def score_texts(texts, max_len=128, batch_size=16):
    """
    Robust FinBERT scorer:
      - hard truncation to max_len (<=512)
      - mini-batches to avoid RAM spikes
    Returns list of (label, confidence) tuples.
    """
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = _tok(
            chunk,
            truncation=True,
            max_length=max_len,   # <<<<<<<<<< hard cap (FinBERT limit is 512)
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = [LABELS[i] for i in probs.argmax(axis=1)]
        scores = probs.max(axis=1)
        out.extend(list(zip(labels, scores)))
    return out

