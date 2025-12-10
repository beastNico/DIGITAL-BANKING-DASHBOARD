import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


def bertopic_cleaning():
    load_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/new_df_clean.parquet")

    if not load_path.exists():
        print("⚠️ new_df_clean.parquet not found. Skipping BERTopic cleaning.")
        return None

    new_df_clean = pd.read_parquet(load_path)
    
    new_df_clean['word_count'] = new_df_clean['review_text'].apply(lambda x: len(str(x).split()))
    print(f"Shape before dropping reviews with less than 4 words: {new_df_clean.shape}")
    new_df_clean = new_df_clean[new_df_clean['word_count'] >= 4].reset_index(drop=True)
    print(f"Shape after dropping reviews with less than 4 words: {new_df_clean.shape}")

    PRAISE_WORDS = [
        "amazing", "awesome", "brilliant", "excellent", "fantastic", "love", "nice", "perfect", "star", "stars",
        "awful", "bad", "disappointing", "horrible", "poor", "terrible", "useless", "worst", "best"
    ]
    CUSTOM_STOP_WORDS = [
        "santander", "revolut", "revolute", "revoult", "revlout", "revelout", "hsbc", "barclays", 
        "barclay", "lloyds","lloyd", "monzo", "app", "banking", "bank"
    ]
    STOP_WORDS = PRAISE_WORDS + CUSTOM_STOP_WORDS

    new_df_clean["bert_review_text"] = new_df_clean['review_text'].astype(str)
    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"http\S+", "", str(x)))
    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    for w in STOP_WORDS:
        new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(rf"\b{re.escape(w)}\b", "", x, flags=re.IGNORECASE))
    new_df_clean["bert_review_text"] = new_df_clean["bert_review_text"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    new_df_clean = new_df_clean[new_df_clean["bert_review_text"].str.split().str.len() >= 2].copy()
    print(f"Shape of new_df_clean after bertopic cleaning: {new_df_clean.shape}")

    new_df_topics = new_df_clean.copy()
    new_df_topics.to_parquet("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/new_df_topics.parquet", index=False)
    return new_df_topics


def apply_bertopic():
    new_df_topics_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/new_df_topics.parquet")
    model_path = Path("D:/Digital-Banking-Dashboard/assets/models/bertopic/seed_final_model")

    if not new_df_topics_path.exists():
        print("⚠️ new_df_topics.parquet not found. Cannot apply BERTopic.")
        return None
    if not model_path.exists():
        print(f"⚠️ BERTopic model folder not found at {model_path}. Cannot apply BERTopic.")
        return None

    new_df_topics = pd.read_parquet(new_df_topics_path)
    bertopic_model = BERTopic.load(str(model_path), embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))

    new_reviews = new_df_topics["bert_review_text"].tolist()
    new_topics, new_probs = bertopic_model.transform(new_reviews)
    new_df_topics["bert_topic"] = new_topics
    new_df_topics["bert_probs"] = new_probs.max(axis=1)
    print(f"Applied BERTopic model. Shape of new_df_topics: {new_df_topics.shape}")

    custom_labels = {
        -1: "Outliers", 0: "Undefined", 1: "Simplicity", 2: "Money Management",
        3: "Usability & Experience", 4: "Security & Close Account", 5: "Login & Authentication",    
        6: "Travel & FX", 7: "Reliability", 8: "Cards", 9: "Customer Service",
        10: "Compatibility & Launch Issues", 11: "Stability", 12: "Layout & Interface",
        13: "Cheque", 14: "Investments & Fees", 15: "Updates", 16: "Functional Bugs",
        17: "Notifications & Ads", 18: "Chat", 19: "Referral Program"
    } 
    new_df_topics['bert_label'] = new_df_topics['bert_topic'].map(custom_labels)

    print(f"Shape before removing 'Outliers' and 'Undefined': {new_df_topics.shape}")
    new_df_topics = new_df_topics[~new_df_topics['bert_label'].isin(['Outliers', 'Undefined'])].reset_index(drop=True)
    print(f"Shape after removing 'Outliers' and 'Undefined': {new_df_topics.shape}")

    macro_labels = {
        "Simplicity": "User Experience", "Usability & Experience": "User Experience",
        "Layout & Interface": "User Experience", "Notifications & Ads": "User Experience",
        "Cards": "Products", "Investments & Fees": "Products", "Cheque": "Products",
        "Referral Program": "Products", "Customer Service": "Customer Service", "Chat": "Customer Service",
        "Reliability": "Performance", "Stability": "Performance", "Compatibility & Launch Issues": "Performance",
        "Updates": "Performance", "Functional Bugs": "Performance"
    }
    new_df_topics['bert_macro_label'] = new_df_topics['bert_label'].map(macro_labels).fillna(new_df_topics['bert_label'])
    new_df_topics['subset'] = 'new reviews'

    cols = ['subset','app','score','review_text','review_date','word_count','bert_macro_label','bert_label','bert_probs','scrape_date']
    new_df_topics = new_df_topics[cols]

    for col in new_df_topics.select_dtypes(include=["datetimetz"]).columns:
        new_df_topics[col] = new_df_topics[col].dt.tz_localize(None)
    new_df_topics.to_excel("D:/Digital-Banking-Dashboard/assets/dfs_pipeline/new_df_topics.xlsx", index=False)
    new_df_topics.to_parquet("D:/Digital-Banking-Dashboard/assets/dfs_pipeline/new_df_topics.parquet", index=False)

    return new_df_topics
