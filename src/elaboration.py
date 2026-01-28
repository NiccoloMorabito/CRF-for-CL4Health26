import sys
from groq import Groq
import json
import pydantic 
from pydantic import BaseModel
from typing import get_args, Literal
import os

from prompts import SYSTEM_PROMPT
from dataset_utils import load_or_download_dataset
from models import build_crf_model

LANGUAGE = "en" #TODO
MODEL_ID = "qwen/qwen3-32b" #TODO with this model it does print also the think for some reasons
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
OUTPUT_FILE = f"{MODEL_ID.split("/")[-1]}_27jan_{LANGUAGE}.jsonl"
OUTPUT_PATH = os.path.join("CRF-filling-CL4Health2026", "your_submission_data", OUTPUT_FILE)

FS_DATASET = "NLP-FBK/dyspnea-crf-train"
NUM_FS_EXAMPLES = 3
MAX_RETRIES = 3

API_KEY = "REDACTED_API_KEY" #TODO move
if not API_KEY: #TODO remove or make it useful
    print("Error: GROQ_API_KEY environment variable not set.")
    print("Please export your API key: export GROQ_API_KEY='your_key_here'")
    sys.exit(1)

def extract_crf(dataset: str): #TODO clean this method
    # one-time initializations
    client = Groq(api_key=API_KEY)
    CRFModel = build_crf_model()

    # Load development dataset
    ds = load_or_download_dataset(dataset)[LANGUAGE]
    print(f"{dataset} has {len(ds)} documents")

    base_messages = create_base_messages(CRFModel)

    # with open(OUTPUT_PATH, "r") as f:
    #     processed_ids = {json.loads(line)["document_id"] for line in f}
    
    with open(OUTPUT_PATH, "w") as f: #TODO this is append now
        for item in ds:
            doc_id = item["document_id"]
            # if doc_id in processed_ids:
            #     print(f"Skipped {doc_id} because already processed")
            #     continue
            clinical_note = item["clinical_note"]
            
            print(f"Processing document: {doc_id}")
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    # 1. Invoke model (Network/API errors cached here)
                    raw_output = invoke_model(client, clinical_note, base_messages.copy())
                    
                    # 2. Parse JSON
                    data_dict = json.loads(raw_output)
                    
                    # 3. Validate with Pydantic
                    result = CRFModel.model_validate(data_dict)
                    
                    # 4. Save
                    save_prediction(f, doc_id, result)
                    success = True
                    break # Exit retry loop on success

                except json.JSONDecodeError as e:
                    print(f"[Attempt {attempt+1}/{MAX_RETRIES}] JSON Error for {doc_id}: {e}")
                    print(f"FAILED OUTPUT:\n{raw_output}\n")
                    # Retry
                
                except pydantic.ValidationError as e:
                    print(f"[Attempt {attempt+1}/{MAX_RETRIES}] Validation Error for {doc_id}:")
                    print(e)
                    print(f"FAILED OUTPUT:\n{raw_output}\n")
                    # Retry
                
                except Exception as e:
                    print(f"[Attempt {attempt+1}/{MAX_RETRIES}] System/API Error for {doc_id}: {e}")
                    # Retry (TODO maybe add sleep here if needed)
                    return # TODO currently stopping in case the tokens for that day are over

            if not success:
                print(f"Skipping document {doc_id} after {MAX_RETRIES} failed attempts.")
                
                # Fallback: Create fake instance with all "unknown"
                fake_instance = CRFModel()
                save_prediction(f, doc_id, fake_instance)

def create_base_messages(model: type[BaseModel]): #TODO I made this method called every time, so it's better how it was before
    # system prompt and expected schema
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nCRF Schema:\n" + build_compact_schema(model)
        }
    ]

    # few-shot examples
    fs_examples = load_or_download_dataset(FS_DATASET)[LANGUAGE]
    for ex in fs_examples.select(range(NUM_FS_EXAMPLES)):
        messages.append({
            "role": "user",
            "content": f"Extract structured data from this clinical note:\n\n{ex["clinical_note"]}"
        })
        # Filter out unknown values to save tokens and match system prompt instructions
        filtered_annotations = {item["item"]: item["ground_truth"] for item in ex["annotations"] if item["ground_truth"] != "unknown"}
        messages.append({
            "role": "assistant",
            "content": json.dumps(filtered_annotations, ensure_ascii=False)
        })
    return messages

def invoke_model(client, clinical_note: str, messages):
    messages.append({
        "role": "user",
        "content": f"""
Now process the following clinical note: 
{clinical_note}
"""
    })

    # Let exceptions propagate to be handled by the caller
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0,# default = 1 TODO change this, especially maybe after errors
    )

    raw_output = response.choices[0].message.content
    return raw_output

def build_compact_schema(model: type[BaseModel], remove_unknown_values: bool = False) -> str:
    lines = []

    for field_name, field in model.model_fields.items():
        annotation = field.annotation

        # This is your case: Literal["y", "n", "unknown", ...]
        if annotation is not None and getattr(annotation, "__origin__", None) is Literal:
            values = get_args(annotation)
            if remove_unknown_values:
                values = [value for value in values if value != "unknown"]
                #NOTE: removing unknown from the possible values here ^ considering the current system prompt
            joined = " | ".join(values)
            lines.append(f"{field_name}: {joined}")

    return "\n".join(lines)

def save_prediction(f, doc_id, result):
    # Format output
    predictions = []
    #TODO
    # Result can be a Pydantic model or a fake_instance (which might be a Pydantic model too if initialized that way)
    # If result is already a dict (unlikely with current logic but good for safety), use it directly.
    # But current logic uses CRFModel instance.
    
    data_dict = result.model_dump()
    
    for field_name, value in data_dict.items():
        predictions.append({"item": field_name, "prediction": value})
    
    output_entry = {
        "document_id": doc_id,
        "predictions": predictions
    }
    
    f.write(json.dumps(output_entry) + "\n")
    f.flush() # Ensure it's written immediately 

if __name__ == "__main__":
    extract_crf("NLP-FBK/dyspnea-crf-development")
