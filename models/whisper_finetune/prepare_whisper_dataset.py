"""
prepare_whisper_dataset.py
--------------------------
Helper script to load the cleaned Dioula dataset and prepare it
for fine-tuning OpenAI Whisper using Hugging Face Transformers.

Requirements (add to requirements.txt):
    transformers
    torch
    datasets

Usage:
    python models/whisper_finetune/prepare_whisper_dataset.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperProcessor

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WHISPER_MODEL = "openai/whisper-small"   # Change to whisper-medium / large as needed
# Whisper doesn't natively support Dioula. We use "french" as a fallback token 
# for the processor. The fine-tuning process will adapt the model to Dioula.
TARGET_LANGUAGE = "french"               
TARGET_TASK = "transcribe"
TARGET_SR = 16000

SPLITS = {
    "train": "data/processed/train.csv",
    "dev":   "data/processed/dev.csv",
    "test":  "data/processed/test.csv",
}

# Base directory of the project (two levels up from this script)
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def load_split(csv_path):
    """Loads a CSV split and returns a Hugging Face Dataset."""
    df = pd.read_csv(csv_path)

    # Use the normalised audio if available, otherwise fall back to raw wav
    def resolve_path(p):
        normalized = p.replace("data/audio/wav", "data/audio/normalized")
        abs_norm = os.path.join(BASE_DIR, normalized)
        abs_raw  = os.path.join(BASE_DIR, p)
        if os.path.exists(abs_norm):
            return abs_norm
        return abs_raw

    df["audio"] = df["audio_path"].apply(resolve_path)

    # Keep only the columns Whisper needs
    df = df[["audio", "dyu_clean"]].rename(columns={"dyu_clean": "transcription"})
    df = df.dropna(subset=["audio", "transcription"])

    dataset = Dataset.from_pandas(df, preserve_index=False)
    # We keep 'audio' as a string path and load it manually with librosa later
    # to avoid FFmpeg/torchcodec dependencies on Windows.
    return dataset


import librosa

def prepare_features(batch, processor):
    """
    Converts raw audio + transcription into Whisper model inputs:
    - input_features : Log-Mel Spectrogram  (model input)
    - labels         : tokenised transcription (target)
    """
    # Load audio manually using librosa to bypass HF datasets audio backend issues
    audio_arrays = [librosa.load(path, sr=TARGET_SR)[0] for path in batch["audio"]]

    # Extract log-mel spectrograms
    batch["input_features"] = processor.feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    ).input_features

    # Tokenise transcriptions
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch


def main():
    print(f"Loading Whisper processor: {WHISPER_MODEL}")
    processor = WhisperProcessor.from_pretrained(
        WHISPER_MODEL,
        language=TARGET_LANGUAGE,
        task=TARGET_TASK,
    )

    # Build DatasetDict
    dataset_dict = DatasetDict()
    for split_name, csv_path in SPLITS.items():
        full_path = os.path.join(BASE_DIR, csv_path)
        if not os.path.exists(full_path):
            print(f"[SKIP] {csv_path} not found.")
            continue
        print(f"Loading split: {split_name} from {csv_path} ...")
        dataset_dict[split_name] = load_split(full_path)

    print("\nDataset loaded:")
    print(dataset_dict)

    # Apply feature extraction (batched for speed)
    print("\nExtracting Whisper features (this may take a while)...")
    dataset_dict = dataset_dict.map(
        lambda batch: prepare_features(batch, processor),
        batched=True,
        batch_size=8,
        remove_columns=["audio", "transcription"],
    )

    print("\nFinal dataset:")
    print(dataset_dict)

    # Optional: save to disk for fast re-loading
    save_path = os.path.join(BASE_DIR, "data", "processed", "whisper_dataset")
    dataset_dict.save_to_disk(save_path)
    print(f"\nDataset saved to: {save_path}")


if __name__ == "__main__":
    main()
