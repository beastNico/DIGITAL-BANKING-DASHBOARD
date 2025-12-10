import pandas as pd
from pathlib import Path


def clean_new_reviews():
    load_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/new_df_raw.parquet")

    # --- SAFETY CHECK ---
    if not load_path.exists():
        print("No new raw reviews found. Skipping cleaning step.")
        return None

    # --- Original code starts here ---
    df = pd.read_parquet(load_path)
    print(f"Original DataFrame shape: {df.shape}")

    df = df.drop(columns=['app_id','reviewId','user_name','thumbs_up','Reply','Reply_Date','App_Version'])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    colmap = {
        'app_name': 'app',
        'text': 'review_text',
        'date': 'review_date',
    }
    new_df_cleaned = df.rename(columns=colmap)

    print(f"Cleaned DataFrame shape: {new_df_cleaned.shape}")

    new_cleaned_path = Path("D:/Digital-Banking-Dashboard/assets/dfs_pipeline/new_df_clean.parquet")
    new_df_cleaned.to_parquet(new_cleaned_path, index=False)
    print(f"Saved cleaned new reviews → {new_cleaned_path}")

    return new_df_cleaned



def update_df_clean():
    existing_df_clean_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/df_clean.parquet")
    new_df_clean_path = Path("D:/Digital-Banking-Dashboard/assets/intermediate_dfs/new_df_clean.parquet")

    # --- SAFETY CHECK: existing df_clean must exist ---
    if not existing_df_clean_path.exists():
        print("ERROR: Existing df_clean.parquet not found! Cannot update.")
        return None

    existing_df_clean = pd.read_parquet(existing_df_clean_path)

    # --- SAFETY CHECK: new cleaned reviews ---
    if not new_df_clean_path.exists():
        print("No new cleaned reviews found. df_clean remains unchanged.")
        return existing_df_clean

    new_df_clean = pd.read_parquet(new_df_clean_path)

    print(f"Existing cleaned DataFrame shape: {existing_df_clean.shape}")
    print(f"New cleaned DataFrame shape: {new_df_clean.shape}")

    updated_df_clean = pd.concat([existing_df_clean, new_df_clean], ignore_index=True)
    print(f"Updated cleaned DataFrame shape: {updated_df_clean.shape}")

    updated_df_clean.to_parquet(existing_df_clean_path, index=False)
    print(f"Updated cleaned DataFrame saved to {existing_df_clean_path}")

    return updated_df_clean
