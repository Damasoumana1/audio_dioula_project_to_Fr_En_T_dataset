import os
import soundfile as sf
from tqdm import tqdm
from datasets import Audio
import librosa
import io
import numpy as np

def extract_audio_files(dataset, split_name, output_dir):
    """
    Extracts audio arrays from the dataset and saves them as .wav files.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []
    
    # Disable automatic decoding to avoid torchcodec requirement on Python 3.13
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    print(f"Extracting audio for {split_name}...")
    for i, item in enumerate(tqdm(dataset[split_name])):
        audio_item = item['audio']
        
        # Manually decode using librosa
        audio_bytes = audio_item['bytes']
        if audio_bytes is None:
            continue
            
        # Load audio from bytes
        with io.BytesIO(audio_bytes) as b:
            audio_array, sampling_rate = librosa.load(b, sr=None)
        
        # Create a filename based on dyu_id or index
        # Strip any existing extension from dyu_id to avoid double extension (e.g. .wav.wav)
        raw_id = os.path.splitext(item['dyu_id'])[0] if 'dyu_id' in item else f"sample_{i}"
        filename = f"{raw_id}.wav"
        file_path = os.path.join(output_dir, filename)
        
        # Save as wav
        sf.write(file_path, audio_array, sampling_rate)
        
        # Keep track for metadata with ALL columns from Hugging Face
        metadata_list.append({
            'dyu_id': item.get('dyu_id', ''),
            'fr_id': item.get('fr_id', ''),
            'audio_path': file_path,
            'dyu': item.get('dyu', ''),
            'fr': item.get('fr', ''),
            'en': item.get('en', ''),
            'gender': item.get('gender', ''),
            'age_group': item.get('age_group', ''),
            'duration': item.get('duration', 0),
            'sampling_rate': sampling_rate,
            'country': item.get('country', ''),
            'commonvoice_split': item.get('commonvoice_split', '')
        })
        
        # Stop early for testing if needed
        # if i > 10: break

    return metadata_list
