import pandas as pd
import os

CSV_FILES = [
    "data/processed/metadata.csv",
    "data/processed/train.csv",
    "data/processed/dev.csv",
    "data/processed/test.csv",
    "data/processed/validation.csv",
]

def fix_paths(csv_path):
    if not os.path.exists(csv_path):
        print(f"[SKIP] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if 'audio_path' not in df.columns:
        print(f"[SKIP] No 'audio_path' column in: {csv_path}")
        return

    before = df['audio_path'].str.contains(r'\.wav\.wav', regex=True).sum()
    df['audio_path'] = df['audio_path'].str.replace('.wav.wav', '.wav', regex=False)
    # Also normalise backslashes to forward slashes for cross-platform consistency
    df['audio_path'] = df['audio_path'].str.replace('\\', '/', regex=False)

    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[OK] Fixed {before} paths in {csv_path}")

if __name__ == "__main__":
    for f in CSV_FILES:
        fix_paths(f)
    print("Path correction complete.")
