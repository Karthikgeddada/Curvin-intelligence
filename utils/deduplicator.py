import pandas as pd

def deduplicate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicates news articles based on URL and Title."""
    if news_df.empty:
        return news_df
    
    # Primary deduplication by URL
    news_df = news_df.drop_duplicates(subset=['link'])
    
    # Secondary deduplication by Title (case-insensitive)
    news_df['title_lower'] = news_df['title'].str.lower().str.strip()
    news_df = news_df.drop_duplicates(subset=['title_lower'])
    news_df = news_df.drop(columns=['title_lower'])
    
    return news_df.reset_index(drop=True)
