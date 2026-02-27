import os
from datasets import load_dataset, load_from_disk

# Collection URL: https://huggingface.co/collections/NLP-FBK/crf-filling-sharedtask-cl4health2026-6752d58a1768411d9961db1d
DATASET_NAMES = [
    "NLP-FBK/synthetic-crf-train",
    "NLP-FBK/dyspnea-crf-train",
    "NLP-FBK/dyspnea-crf-development",
    "NLP-FBK/dyspnea-clinical-notes",
    "NLP-FBK/dyspnea-valid-options",
    "NLP-FBK/dyspnea-crf-test",
]

def get_dataset_path(dataset_name: str) -> str:
    local_name = dataset_name.split('/')[-1]
    save_path = os.path.join("data", local_name)
    return save_path

def load_or_download_dataset(dataset_name: str):
    save_path = get_dataset_path(dataset_name)
    if os.path.exists(save_path):
        print(f"Found local cache at {save_path}, loading from disk...")
        ds = load_from_disk(save_path)
    else:
        print(f"Not found locally, downloading from Hugging Face...")
        ds = load_dataset(dataset_name)
        ds.save_to_disk(save_path)
        print(f"Saved to {save_path}")
    return ds

def download_and_save_datasets(): #TODO this could also be deleted probably
    print("Starting dataset processing...")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    for name in DATASET_NAMES:
        print(f"\nProcessing {name}...")
        
        try:
            ds = load_or_download_dataset(name)
            
            # Print some info to verify
            print(f"Successfully loaded {name}")
            print(f"Structure: {ds}")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")


if __name__ == "__main__":
    # download_and_save_datasets()
    ds = load_or_download_dataset("NLP-FBK/dyspnea-crf-test")["en"]
    print(ds)
    for el in ds:
        print(len(el["annotations"]))
