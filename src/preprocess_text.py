import re
import pandas as pd
import os

# Dioula-specific characters to preserve (tonal vowels, nasal characters)
# We keep: letters, digits, spaces, apostrophes, dioula extended chars
KEEP_PATTERN = re.compile(r"[^\w\s\'\u0250-\u02AF\u1D00-\u1DBF\u1E00-\u1EFF]", re.UNICODE)

CSV_FILES = [
    "data/processed/metadata.csv",
    "data/processed/train.csv",
    "data/processed/dev.csv",
    "data/processed/test.csv",
]

def clean_text(text):
    """
    Basic text cleaning for Dioula/French.
    Lowercases, strips extra whitespace, removes punctuation
    while preserving Dioula-specific extended Latin characters.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower().strip()
    text = KEEP_PATTERN.sub("", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_metadata(df):
    """
    Applies cleaning to the dyu and fr columns of a dataframe.
    Adds dyu_clean and fr_clean columns.
    """
    df["dyu_clean"] = df["dyu"].apply(clean_text)
    df["fr_clean"] = df["fr"].apply(clean_text)
    return df


def apply_cleaning_to_all_csvs():
    """
    Loads each split CSV, applies text cleaning, and saves in-place.
    """
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"[SKIP] Not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = preprocess_metadata(df)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] Cleaned transcriptions in {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    apply_cleaning_to_all_csvs()
    print("Text preprocessing complete.")
