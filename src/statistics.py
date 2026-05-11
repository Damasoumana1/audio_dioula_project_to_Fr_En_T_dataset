import pandas as pd
import os

def generate_dataset_report(metadata_path, output_path):
    """
    Generates a basic statistical report of the dataset.
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found.")
        return

    df = pd.read_csv(metadata_path)
    
    report = []
    report.append("=== Dataset Statistics ===")
    report.append(f"Total recordings: {len(df)}")
    report.append(f"Total duration (hours): {df['duration'].sum() / 3600:.2f}")
    report.append(f"Average duration (sec): {df['duration'].mean():.2f}")
    
    if 'gender' in df.columns:
        report.append("\nGender Distribution:")
        report.append(df['gender'].value_counts().to_string())
        
    if 'age_group' in df.columns:
        report.append("\nAge Group Distribution:")
        report.append(df['age_group'].value_counts().to_string())

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in report])
    
    print(f"Report generated at {output_path}")

if __name__ == "__main__":
    generate_dataset_report("data/processed/metadata.csv", "outputs/reports/dataset_report.txt")
