import json
from typing import get_args, Literal
from pydantic import BaseModel

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
