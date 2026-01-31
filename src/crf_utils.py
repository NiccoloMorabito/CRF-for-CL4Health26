from google.genai.types import GenerateContentConfig
from groq import Groq
from openai import OpenAI
import json
from typing import get_args, Literal, Optional, Any
from pydantic import BaseModel
import google.genai as genai
from google.genai import types

from dataset_utils import load_or_download_dataset

# LLMType defines the supported model families:
# - groq: Groq API (using text-based schema enforcement via prompt)
# - gemma: Google GenAI SDK using Gemma models (text-based schema)
# - gemini: Google GenAI SDK using Gemini models (native structured output)
# - openai: OpenAI API (native structured output)
LLMType = Literal["groq", "gemma", "gemini", "openai"]

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

def create_base_messages(
    model: type[BaseModel], 
    system_prompt: str, 
    fs_dataset_name: str, 
    language: str, 
    num_fs_examples: int, 
    llm_type: LLMType
) -> tuple[list, Optional[str]]:
    """
    Creates the base messages for the LLM.
    Returns:
        messages: A list of message dictionaries (or Content objects for GenAI).
        system_instruction: The system prompt string, if valid for the LLM usage (e.g. Gemini/OpenAI),
                            or None if it was embedded in the messages (Groq/Gemma).
    """
    answer_role = "assistant"
    use_native_schema = False
    
    # 1. Configuration based on LLM Type
    if llm_type == "groq":
        answer_role = "assistant"
        use_native_schema = False # Text-based schema (but we will enforce JSON object mode)
    elif llm_type == "gemma":
        answer_role = "model"
        use_native_schema = False # Text-based schema, no system config
    elif llm_type == "gemini":
        answer_role = "model"
        use_native_schema = True # Native schema, system config supported
    elif llm_type == "openai":
        answer_role = "assistant"
        use_native_schema = True # Native schema
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    
    messages = []
    returned_system_instruction = None

    # 2. System Prompt & Schema Logic
    if use_native_schema:
        # Native Schema: Schema is NOT appended to text. System prompt is passed separately or cleanly.
        
        if llm_type == "openai":
            # OpenAI: System message in history, native schema via response_format
            messages.append({"role": "system", "content": system_prompt})
        elif llm_type == "gemini":
            # Gemini: System prompt passed via config (system_instruction)
            returned_system_instruction = system_prompt
            # No system message in history list
    else:
        # Text-based Schema: Append schema to system prompt content
        full_system_content = system_prompt + "\n\nCRF Schema:\n" + build_compact_schema(model)
        
        if llm_type == "groq":
             messages.append({
                "role": "system", 
                "content": full_system_content
            })
        elif llm_type == "gemma":
            # Gemma (GenAI): No system config. Usually passed as "user" message at start.
             messages.append({
                "role": "user", 
                "content": full_system_content
            })
    
    # 3. Few-shot Examples
    fs_examples = load_or_download_dataset(fs_dataset_name)[language]
    for ex in fs_examples.select(range(num_fs_examples)):
        messages.append({
            "role": "user",
            "content": f"Extract structured data from this clinical note:\n\n{ex['clinical_note']}"
        })
        # Filter unknown values for non-native schema (to save tokens)
        # For native schema, we could technically include them, but clean data is better.
        filtered_annotations = {item["item"]: item["ground_truth"] for item in ex["annotations"] if item["ground_truth"] != "unknown"}
        messages.append({
            "role": answer_role,
            "content": json.dumps(filtered_annotations, ensure_ascii=False)
        })

    # 4. Format Conversion for GenAI (Gemma/Gemini)
    if llm_type in ["gemma", "gemini"]:
        # Convert dicts to types.Content
        new_messages = [
            types.Content(role=mex["role"], parts=[types.Part(text=mex["content"])]) 
            for mex in messages
        ]
        messages = new_messages

    return messages, returned_system_instruction


def invoke_model(
    client: Any, 
    model_id: str, 
    clinical_note: str, 
    messages: list, 
    attempt: int,
    system_instruction: Optional[str] = None,
    pydantic_model: Optional[type[BaseModel]] = None
) -> Union[str, BaseModel]:
    """
    Invokes the LLM with the given messages and configuration.
    
    Args:
        client: The initialized API client (Groq, OpenAI, or genai.Client).
        model_id: The model identifier string.
        clinical_note: The input text to process.
        messages: The context messages (history).
        attempt: The retry attempt number (used for temperature scaling).
        system_instruction: Optional system prompt (used for Gemini).
        pydantic_model: Optional Pydantic model for structured output enforcement (OpenAI/Gemini).
    
    Returns:
        Union[str, BaseModel]: raw JSON string (Groq/Gemma) OR validated Pydantic model (OpenAI/Gemini).
    """
    
    final_message_text = f"Now process the following clinical note:\n{clinical_note}"
    
    temperature = attempt * 0.2 

    # --- GROQ ---
    if isinstance(client, Groq):
        messages.append({
            "role": "user",
            "content": final_message_text
        })
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_format="hidden" if "deepseek" in model_id.lower() or "qwen" in model_id.lower() else None
        )
        return response.choices[0].message.content

    # --- OPENAI ---
    elif isinstance(client, OpenAI):
        messages.append({
            "role": "user",
            "content": final_message_text
        })
        
        if pydantic_model:
            response = client.beta.chat.completions.parse(
                model=model_id,
                messages=messages,
                # temperature=temperature, TODO removed because gpt-5-nano doesn't support it
                response_format=pydantic_model
            )
            # Return validating parsed object directly
            if response.choices[0].message.parsed:
                 return response.choices[0].message.parsed
            elif response.choices[0].message.refusal:
                 raise ValueError(f"Model refused to answer: {response.choices[0].message.refusal}")
            else:
                 return response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content

    # --- GOOGLE GENAI (Gemini / Gemma) ---
    elif isinstance(client, genai.Client):
        messages.append(types.Content(role="user", parts=[types.Part(text=final_message_text)]))
        
        config_args = {"temperature": temperature * 0.5}
        
        is_gemini = "gemini" in model_id
        
        if is_gemini:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = pydantic_model
            if system_instruction:
                config_args["system_instruction"] = system_instruction
        
        # Gemma uses default config (no system instruction, no schema enforcement in config)
        
        response = client.models.generate_content(
            model=model_id,
            contents=messages,
            config=GenerateContentConfig(**config_args),
        )
        
        # If Gemini/Structured, we parse and return the object
        if is_gemini and pydantic_model:
             try:
                 # GenAI usually returns a JSON string in text. We validate it manually into the Pydantic model.
                 return pydantic_model.model_validate_json(response.text)
             except Exception as e:
                 # Fallback if parsing fails (unlikely with response_schema)
                 return response.text
        
        return response.text

    else:
        raise ValueError(f"Unknown LLM type: {type(client)}")

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
