from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def fetch_intent_data(labels=[0, 1], samples_per_label=100, test_size=0.2):
    """
    Fetches the Tanaos Intent dataset from Hugging Face.
    Includes auto-detection for column names and version-proof sampling.
    """
    print(f"📡 Requesting dataset: tanaos/synthetic-intent-classifier-dataset-v1...")

    # Load the dataset
    raw_dataset = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1", split='train')
    df = raw_dataset.to_pandas()

    # Auto-detect and standardize the classification column to 'label'
    if 'intent' in df.columns:
        df = df.rename(columns={'intent': 'label'})
    elif 'labels' in df.columns:
        df = df.rename(columns={'labels': 'label'})

    # Failsafe
    if 'label' not in df.columns:
        raise KeyError(f"Critical Error: Could not find classification column. Available: {list(df.columns)}")

    # Filter for our target intents (0: greeting, 1: farewell)
    df_filtered = df[df['label'].isin(labels)].copy()

    # Version-Proof Class Balancing (Avoids the groupby index trap)
    balanced_frames = []
    for _, group in df_filtered.groupby('label'):
        sampled_group = group.sample(n=min(len(group), samples_per_label), random_state=42)
        balanced_frames.append(sampled_group)

    # Reassemble the dataframe safely
    balanced_df = pd.concat(balanced_frames).reset_index(drop=True)

    # Split into training and testing sets
    train_df, test_df = train_test_split(
        balanced_df, 
        test_size=test_size, 
        stratify=balanced_df['label'], 
        random_state=42
    )

    print(f"✅ Success: {len(train_df)} training and {len(test_df)} testing samples retrieved.")
    return train_df, test_df
