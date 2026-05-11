import re

def clean_text(text):
    """
    Basic text cleaning for Dioula/French.
    """
    if not text:
        return ""
    # Lowercase, remove extra spaces, remove specific punctuation
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', '', text)
    return text

def preprocess_metadata(df):
    """
    Applies cleaning to the entire metadata dataframe.
    """
    df['dyu_clean'] = df['dyu'].apply(clean_text)
    df['fr_clean'] = df['fr'].apply(clean_text)
    return df

if __name__ == "__main__":
    import pandas as pd
    # Example usage
    # df = pd.read_csv("data/processed/metadata.csv")
    # df = preprocess_metadata(df)
    # df.to_csv("data/processed/cleaned_metadata.csv", index=False)
