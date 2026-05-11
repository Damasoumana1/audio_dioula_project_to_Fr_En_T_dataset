from datasets import load_dataset
import os

def download_dataset(repo_id="uvci/koumankan4dyula"):
    """
    Downloads the dataset from Hugging Face.
    """
    print(f"Downloading dataset {repo_id}...")
    dataset = load_dataset(repo_id)
    print("Download complete.")
    return dataset

if __name__ == "__main__":
    ds = download_dataset()
    print(ds)
