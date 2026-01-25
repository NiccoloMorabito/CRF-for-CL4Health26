from typing import Literal, get_args
from pydantic import BaseModel, Field, create_model
from dataset_utils import load_or_download_dataset

def build_crf_model() -> type[BaseModel]:
    dataset = load_or_download_dataset("NLP-FBK/dyspnea-valid-options")["train"]
    print(dataset)
    fields = {}

    for row in dataset:
        item = row["item"]
        options = tuple(row["options"])  # must be tuple for Literal

        fields[item] = (
            Literal[options],  # type
            Field(default="unknown")  # safe default
        )

    return create_model(
        "DyspneaCRF",
        **fields,
        __base__=BaseModel,
    )

if __name__=='__main__':
    model = build_crf_model()
    print(model)
    schema = model.model_json_schema()
    from pprint import pprint
    pprint(schema)
