import sys
from groq import Groq
import json
from prompts import SYSTEM_PROMPT
from dataset_utils import load_or_download_dataset
from models import build_crf_model
from pydantic import BaseModel
from typing import get_args, Literal
import os

LANGUAGE = "en" #TODO
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
OUTPUT_FILE = f"{MODEL_ID.split("/")[-1]}_retrying-on-errors.jsonl"
OUTPUT_PATH = os.path.join("CRF-filling-CL4Health2026", "your_submission_data", OUTPUT_FILE)

API_KEY = "REDACTED_API_KEY" #TODO move
if not API_KEY:
    print("Error: GROQ_API_KEY environment variable not set.")
    print("Please export your API key: export GROQ_API_KEY='your_key_here'")
    sys.exit(1)

import pydantic 

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


def create_base_messages(): #TODO I made this method called every time, so it's better how it was before
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Inject few-shot examples
    gold_standard_pairs = load_or_download_dataset("NLP-FBK/dyspnea-crf-train")[LANGUAGE]
    for ex in gold_standard_pairs.select(range(5)):
        messages.append({
            "role": "user",
            "content": f"Extract structured data from this clinical note:\n\n{ex["clinical_note"]}"
        })
        # Filter out unknown values to save tokens and match system prompt instructions
        filtered_annotations = {item["item"]: item["ground_truth"] for item in ex["annotations"] if item["ground_truth"] != "unknown"}
        # print(filtered_annotations)
        messages.append({
            "role": "assistant",
            "content": json.dumps(filtered_annotations, ensure_ascii=False)
        })
    return messages


def build_compact_schema(model: type[BaseModel]) -> str:
    lines = [
        "All fields are strings.",
        "For each field, output EXACTLY one of the allowed values.",
        ""
    ]

    for field_name, field in model.model_fields.items():
        annotation = field.annotation

        # This is your case: Literal["y", "n", "unknown", ...]
        if annotation is not None and getattr(annotation, "__origin__", None) is Literal:
            values = get_args(annotation)
            joined = " | ".join(values)
            lines.append(f"{field_name}: {joined}")

    return "\n".join(lines)

def invoke_model(client, clinical_note: str, CRFModel):
    messages = create_base_messages()

    messages.append({
        "role": "user",
        "content": f"""
Extract structured data from the following clinical note.
Follow this JSON schema exactly:
{build_compact_schema(CRFModel)}

Clinical note:
{clinical_note}
"""
    })

    # Let exceptions propagate to be handled by the caller
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        # temperature=0, default = 1 TODO change this, especially maybe after errors
    )

    raw_output = response.choices[0].message.content
    return raw_output


def crf_dataset(): #TODO clean this method
    # one-time initializations
    client = Groq(api_key=API_KEY)
    CRFModel = build_crf_model()

    # Load development dataset
    ds = load_or_download_dataset("NLP-FBK/dyspnea-crf-development")[LANGUAGE]
    print(len(ds))
    
    with open(OUTPUT_PATH, "w") as f:
        for item in ds:
            doc_id = item["document_id"]
            clinical_note = item["clinical_note"]
            
            print(f"Processing document: {doc_id}")
                        
            MAX_RETRIES = 3
            success = False

            for attempt in range(MAX_RETRIES):
                try:
                    # 1. Invoke model (Network/API errors cached here)
                    raw_output = invoke_model(client, clinical_note, CRFModel)
                    
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

            if not success:
                print(f"Skipping document {doc_id} after {MAX_RETRIES} failed attempts.")
                
                # Fallback: Create fake instance with all "unknown"
                fake_instance = CRFModel()
                save_prediction(f, doc_id, fake_instance)

                

if __name__ == "__main__":
    crf_dataset()
