import json
from typing import get_args, Literal
from pydantic import BaseModel

from dataset_utils import load_or_download_dataset

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

def create_base_messages(model: type[BaseModel], system_prompt: str, fs_dataset_name: str, language: str, num_fs_examples: int):
    # system prompt and expected schema
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\nCRF Schema:\n" + build_compact_schema(model)
        }
    ]

    # few-shot examples
    fs_examples = load_or_download_dataset(fs_dataset_name)[language]
    for ex in fs_examples.select(range(num_fs_examples)):
        messages.append({
            "role": "user",
            "content": f"Extract structured data from this clinical note:\n\n{ex['clinical_note']}"
        })
        # Filter out unknown values to save tokens and match system prompt instructions
        filtered_annotations = {item["item"]: item["ground_truth"] for item in ex["annotations"] if item["ground_truth"] != "unknown"}
        messages.append({
            "role": "assistant",
            "content": json.dumps(filtered_annotations, ensure_ascii=False)
        })
    return messages

def invoke_model(client, model_id: str, clinical_note: str, messages: list, attempt: int):
    messages.append({
        "role": "user",
        "content": f"""
Now process the following clinical note: 
{clinical_note}
"""
    })

    temperature = attempt * 0.2 # NOTE first shy attempt to make the model escape the same error over and over

    # Let exceptions propagate to be handled by the caller
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,# default = 1
        reasoning_format="hidden" if model_id == "qwen/qwen3-32b" else None
    )

    raw_output = response.choices[0].message.content
    return raw_output

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
