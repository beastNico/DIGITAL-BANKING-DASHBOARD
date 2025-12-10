import pandas as pd
from pathlib import Path


def update_df_monthly():

    """
    Create/Update df_monthly DataFrame aggregating reviews by month and app.
    """

    # Load cleaned DataFrame
    clean_df_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/df_clean.parquet")
    if not clean_df_path.exists():
        print(f"⚠️ File not found: {clean_df_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df_clean = pd.read_parquet(clean_df_path)

    # Convert 'review_date' to datetime if not already
    df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')

    # Extract year-month for aggregation
    df_clean['period_month'] = df_clean['review_date'].dt.to_period('M').dt.to_timestamp('M')
   
    # Aggregate reviews by month and app
    selected = df_clean[['app','period_month','score']].copy()
    selected['app'] = selected['app'].astype('category')

    df_monthly = (selected
        .groupby(['period_month','app'], observed=True)['score']
        .agg(avg_score='mean', n_reviews='size')
        .reset_index()
    )

    # Save df_monthly to Parquet
    monthly_df_path = Path("D:/Digital-Banking-Dashboard/assets/df_monthly.parquet")
    df_monthly.to_parquet(monthly_df_path, index=False)
    print(f"✅ Saved monthly aggregated DataFrame → {monthly_df_path}")

    return df_monthly


def update_df_topic():
    
    """
    Updates the existing DataFrame with topics by appending new reviews with topics.
    """

    existing_df_topic_path = Path("D:/Digital-Banking-Dashboard/assets/df_topic.parquet")
    new_df_topics_path = Path("D:/Digital-Banking-Dashboard/assets/dfs_pipeline/new_df_topics.parquet")

    # Check if files exist
    if not existing_df_topic_path.exists():
        print(f"⚠️ File not found: {existing_df_topic_path}. Returning empty DataFrame.")
        existing_df_topic = pd.DataFrame()
    else:
        existing_df_topic = pd.read_parquet(existing_df_topic_path)

    if not new_df_topics_path.exists():
        print(f"⚠️ File not found: {new_df_topics_path}. Returning empty DataFrame.")
        new_df_topics = pd.DataFrame()
    else:
        new_df_topics = pd.read_parquet(new_df_topics_path)

    if existing_df_topic.empty and new_df_topics.empty:
        print("ℹ️ Both existing and new topics DataFrames are empty. Nothing to update.")
        return pd.DataFrame()

    # print shapes before concatenation
    print(f"Existing df_topic shape: {existing_df_topic.shape}")
    print(f"New df_topics shape: {new_df_topics.shape}")

    # concatenate DataFrames
    updated_df_topic = pd.concat([existing_df_topic, new_df_topics], ignore_index=True)

    # save updated DataFrame
    updated_df_topic.to_parquet(existing_df_topic_path, index=False)

    print(f"✅ Updated df_topic saved to {existing_df_topic_path}. New shape: {updated_df_topic.shape}")

    return updated_df_topic
