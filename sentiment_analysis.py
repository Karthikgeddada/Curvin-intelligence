import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger("sentiment_analysis")

# ── Module-level cached model loader ──────────────────────────────────────────
# @st.cache_resource ensures the model is loaded only ONCE across all sessions,
# preventing repeated 500MB+ downloads and ensuring fast inference.
@st.cache_resource(show_spinner="⚙️ Loading FinBERT model (first run only)...")
def _load_finbert():
    """Loads and caches ProsusAI/finbert tokenizer and model weights."""
    model_name = "ProsusAI/finbert"
    logger.info(f"Loading FinBERT from HuggingFace: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set to inference mode
    return tokenizer, model


class SentimentAnalyzer:
    """
    Enterprise-grade FinBERT sentiment engine.
    Processes financial text with 512-token truncation and returns
    sentiment label + confidence score per article.
    """

    # FinBERT label mapping: 0 → Positive, 1 → Negative, 2 → Neutral
    LABELS = {0: "Positive", 1: "Negative", 2: "Neutral"}
    MAX_TOKENS = 512

    def __init__(self):
        self.tokenizer, self.model = _load_finbert()

    def _predict(self, text: str) -> dict:
        """Runs a single inference pass and returns label + confidence."""
        inputs = self.tokenizer(
            text[:2000],          # Hard char cap to prevent OOM before tokenization
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_TOKENS,
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=1).squeeze()

        label_id = int(torch.argmax(scores).item())
        return {
            "sentiment": self.LABELS[label_id],
            "confidence": round(float(scores[label_id].item()), 4),
        }

    def analyze(self, texts: list) -> pd.DataFrame:
        """
        Processes a list of article strings through FinBERT.
        Returns a DataFrame with 'sentiment' and 'confidence' columns.
        """
        if not texts:
            return pd.DataFrame(columns=["sentiment", "confidence"])

        results = []
        for text in texts:
            try:
                results.append(self._predict(str(text)))
            except Exception as e:
                logger.warning(f"Prediction failed for one article: {e}")
                results.append({"sentiment": "Neutral", "confidence": 0.0})

        return pd.DataFrame(results)


def apply_sentiment_to_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the news DataFrame with FinBERT sentiment scores.
    Uses the 'full_text' column (title + description) for analysis.
    """
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze(news_df["full_text"].tolist())
    return pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)
