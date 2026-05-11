import librosa
import soundfile as sf
import os
from tqdm import tqdm

def normalize_audio(input_dir, output_dir):
    """
    Normalizes audio files (e.g., peak normalization, resampling).
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Normalizing {len(files)} audio files...")
    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        y, sr = librosa.load(input_path, sr=None)
        
        # Simple peak normalization
        y_norm = librosa.util.normalize(y)
        
        sf.write(output_path, y_norm, sr)

if __name__ == "__main__":
    normalize_audio("data/audio/wav", "data/audio/normalized")
