import sys
import pandas as pd
import soundfile as sf
import os

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

print("=== VERIFICATION DU DATASET ===\n")

# 1. Check CSV columns and sample values
for split in ["train", "dev", "test"]:
    csv_path = f"data/processed/{split}.csv"
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    has_clean = "dyu_clean" in df.columns and "fr_clean" in df.columns
    path_ok = not df["audio_path"].str.contains(r"\.wav\.wav", regex=True).any()
    sr_ok = (df["sampling_rate"] == 16000).all()
    print(f"[{split}.csv]")
    print(f"  Lignes       : {len(df)}")
    print(f"  Chemins OK   : {path_ok}  (pas de .wav.wav)")
    print(f"  SR = 16000   : {sr_ok}")
    print(f"  Cols nettoyees: {has_clean}  (dyu_clean, fr_clean)")
    print(f"  Exemple audio: {df['audio_path'].iloc[0]}")
    print(f"  Exemple dyu  : {df['dyu'].iloc[0]}")
    print(f"  Exemple dyu_c: {df['dyu_clean'].iloc[0]}")
    print()

# 2. Check normalized audio folder
norm_dir = "data/audio/normalized"
wav_files = [f for f in os.listdir(norm_dir) if f.endswith(".wav")]
print(f"[Dossier normalise] {len(wav_files)} fichiers WAV")

if wav_files:
    sample_path = os.path.join(norm_dir, wav_files[0])
    info = sf.info(sample_path)
    print(f"  Echantillon  : {wav_files[0]}")
    print(f"  Frequence    : {info.samplerate} Hz")
    print(f"  Duree        : {info.duration:.2f}s")
    print(f"  Canaux       : {info.channels}")

print("\n=== VERIFICATION TERMINEE ===")
