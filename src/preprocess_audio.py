import librosa
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm

TARGET_SR = 16000  # Whisper requires 16 kHz

CSV_FILES = [
    "data/processed/metadata.csv",
    "data/processed/train.csv",
    "data/processed/dev.csv",
    "data/processed/test.csv",
]

def normalize_audio(input_dir, output_dir):
    """
    Loads each WAV file, resamples to TARGET_SR (16 kHz),
    applies peak normalisation, and saves to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    print(f"Processing {len(files)} audio files -> {TARGET_SR} Hz + normalisation...")
    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load & resample to 16 kHz
        y, sr = librosa.load(input_path, sr=TARGET_SR)

        # Peak normalisation (values in [-1, 1])
        y_norm = librosa.util.normalize(y)

        sf.write(output_path, y_norm, TARGET_SR)

    print(f"Done. {len(files)} files saved to '{output_dir}'.")


def update_metadata_paths(input_dir, output_dir):
    """
    Updates audio_path and sampling_rate columns in all metadata CSVs
    to point to the normalised files at TARGET_SR.
    """
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'audio_path' not in df.columns:
            continue

        # Replace path prefix: data/audio/wav → data/audio/normalized
        df['audio_path'] = df['audio_path'].str.replace(
            input_dir.replace('\\', '/'),
            output_dir.replace('\\', '/'),
            regex=False
        )
        df['sampling_rate'] = TARGET_SR
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[OK] Updated paths in {csv_path}")


if __name__ == "__main__":
    input_dir = "data/audio/wav"
    output_dir = "data/audio/normalized"

    normalize_audio(input_dir, output_dir)
    update_metadata_paths(input_dir, output_dir)
    print("Audio preprocessing complete.")
