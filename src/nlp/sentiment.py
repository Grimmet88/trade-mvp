# src/nlp/sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_MODEL = "yiyanghkust/finbert-tone"
# Determine the device (GPU if available, otherwise CPU)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once globally
_tok = AutoTokenizer.from_pretrained(_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL).to(_DEVICE) # Move model to device
_model.eval()

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def score_texts(texts: list[str], max_len: int = 128, batch_size: int = 16) -> list[float]:
    """
    Scores texts using FinBERT.
    
    Returns a list of continuous sentiment scores: P(Positive) - P(Negative).
    """
    if not texts:
        return []
        
    scores_out = []
    
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        
        # 1. Tokenize and move tensors to the correct device
        enc = _tok(
            chunk,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(_DEVICE) # Move all tensors in the dictionary to the device

        with torch.no_grad():
            logits = _model(**enc).logits
        
        # 2. Calculate Softmax probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # 3. Calculate continuous score: P(Positive) - P(Negative)
        # Assuming LABELS indices 0=Neg, 1=Neu, 2=Pos (standard FinBERT output)
        if probs.shape[1] == 3:
            p_neg = probs[:, 0]
            p_pos = probs[:, 2]
            
            # Continuous score: P(Pos) - P(Neg)
            continuous_scores = p_pos - p_neg
        else:
            # Fallback or error: return 0.0 for non-standard output
            continuous_scores = np.zeros(probs.shape[0])
            logging.warning("FinBERT output shape is not 3. Check model labels.")

        scores_out.extend(continuous_scores.tolist())

    logging.info(f"Scored {len(texts)} texts using {_DEVICE} with batch size {batch_size}.")
    return scores_out
