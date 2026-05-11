import os
from src.load_dataset import download_dataset
from src.extract_audio import extract_audio_files
from src.export_metadata import save_metadata, save_transcripts

def main():
    # 1. Download dataset
    ds = download_dataset("uvci/koumankan4dyula")
    
    # 2. Define splits to process
    # The dataset might have train, validation, test. 
    # Let's check keys and process accordingly.
    splits = ds.keys()
    print(f"Available splits: {list(splits)}")
    
    all_metadata = []
    
    for split in splits:
        output_wav_dir = f"data/audio/wav"
        
        # Extract audio and get metadata
        # Note: In a production scenario, you might want to split into folders data/audio/wav/train etc.
        # But your schema shows data/audio/wav/ as a flat directory for all wavs? 
        # Actually, let's follow your data/raw/train/ etc. for split-specific metadata.
        
        metadata_list = extract_audio_files(ds, split, output_wav_dir)
        all_metadata.extend(metadata_list)
        
        # Save split-specific metadata
        split_metadata_path = f"data/processed/{split}.csv"
        save_metadata(metadata_list, split_metadata_path)
    
    # 3. Save global metadata
    save_metadata(all_metadata, "data/processed/metadata.csv")
    
    # 4. Save individual transcript files
    print("Saving transcripts...")
    save_transcripts(all_metadata, "data/transcripts")
    
    print("Full extraction and processing complete!")

if __name__ == "__main__":
    main()
