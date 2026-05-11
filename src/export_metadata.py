import pandas as pd
import os

def save_metadata(metadata_list, output_path):
    """
    Saves the list of metadata dictionaries to a CSV file.
    """
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Metadata saved to {output_path}")

def save_transcripts(metadata_list, base_dir):
    """
    Saves transcripts into individual text files per language.
    """
    dyula_dir = os.path.join(base_dir, 'dyula')
    french_dir = os.path.join(base_dir, 'french')
    english_dir = os.path.join(base_dir, 'english')
    
    os.makedirs(dyula_dir, exist_ok=True)
    os.makedirs(french_dir, exist_ok=True)
    os.makedirs(english_dir, exist_ok=True)
    
    for item in metadata_list:
        base_name = os.path.basename(item['audio_path']).replace('.wav', '.txt')
        
        with open(os.path.join(dyula_dir, base_name), 'w', encoding='utf-8') as f:
            f.write(item['dyu'])
        with open(os.path.join(french_dir, base_name), 'w', encoding='utf-8') as f:
            f.write(item['fr'])
        with open(os.path.join(english_dir, base_name), 'w', encoding='utf-8') as f:
            f.write(item['en'])
