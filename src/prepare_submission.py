from dataset_utils import load_or_download_dataset
import json

def dummy_worst_submission():
    ds = load_or_download_dataset("NLP-FBK/dyspnea-crf-development")["en"]
    result = [{"document_id": el["document_id"], "predictions": [{"item": item["item"], "prediction": "unknown"} for item in el["annotations"]]} for el in ds]

    with open("dummy_worst_submission.jsonl", "w") as f:
        for el in result:
            f.write(json.dumps(el) + "\n")

def dummy_perfect_submission():
    ds = load_or_download_dataset("NLP-FBK/dyspnea-crf-development")["en"]
    result = [{"document_id": el["document_id"], "predictions": [{"item": item["item"], "prediction": item["ground_truth"]} for item in el["annotations"]]} for el in ds]
    # TODO check the key name (id or document_id?)
    print(result)
    
    with open("dummy_perfect_submission.jsonl", "w") as f:
        for el in result:
            f.write(json.dumps(el) + "\n")

"""
def remove_language_from_id():
    filename = "actual_submission.jsonl"
    with open(filename, "r") as f:
        result = [json.loads(line) for line in f]
    
    for el in result:
        el["document_id"] = el["document_id"].split("_")[0]
    
    with open(filename, "w") as f:
        for el in resultxw:
            f.write(json.dumps(el) + "\n")
"""


if __name__ == "__main__":
    # dummy_worst_submission()
    # dummy_perfect_submission()
    remove_language_from_id()