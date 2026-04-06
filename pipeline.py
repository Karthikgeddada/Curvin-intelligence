import streamlit as st
import pandas as pd
from fetch_news import fetch_industry_news
from sentiment_analysis import apply_sentiment_to_news
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("pipeline")

def run_pipeline(industry: str, max_articles: int, region: str = "United States") -> pd.DataFrame | None:
    """
    Orchestrates the full data pipeline:
    1. Fetch news via Google News RSS
    2. Deduplicate + Clean articles
    3. Score sentiment via FinBERT
    4. Return enriched DataFrame
    """
    logger.info(f"Starting pipeline for industry='{industry}', region='{region}', max_articles={max_articles}")
    
    # ── Stage 1: News Ingestion ───────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {max_articles} articles for **{industry}** in **{region}**..."):
        try:
            news_df = fetch_industry_news(industry, max_articles, region)
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
            st.error(f"❌ Failed to fetch news: {e}")
            return None

    if news_df.empty:
        st.warning("⚠️ No articles found for the selected industry and time range.")
        return None
    
    actual_count = len(news_df)
    logger.info(f"Fetched {actual_count} unique articles after deduplication.")

    # ── Stage 2: Sentiment Analysis ───────────────────────────────────────────
    with st.spinner(f"🧠 Running FinBERT sentiment analysis on {actual_count} articles..."):
        try:
            enriched_df = apply_sentiment_to_news(news_df)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            st.error(f"❌ Sentiment engine error: {e}")
            return None

    logger.info("Pipeline complete. Returning enriched DataFrame.")
    return enriched_df
