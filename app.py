import logging
from typing import Literal, Optional

import gradio as gr
import pandas as pd
import torch
from outlines import generate, models, samplers
from pydantic import BaseModel, ConfigDict
from transformers import BitsAndBytesConfig

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model response schema
class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: Literal["0", "1"]


# Cached model loading
def get_outlines_model(model_id: str, device_map: str = "auto", quantization_bits: Optional[int] = 4):
    if quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model = models.transformers(
        model_id,
        model_kwargs={"device_map": device_map, "quantization_config": quantization_config},
        tokenizer_kwargs={"clean_up_tokenization_spaces": True},
    )
    return model


# Inference
def label_responses(story, question, criteria, response_file):
    df = pd.read_csv(response_file.name)
    assert "response" in df.columns, "CSV must contain a 'response' column."

    # Build prompts
    prompts = [
        f"Story:\n{story}\n\nQuestion:\n{question}\n\nResponse:\n{resp}\n\nCriteria:\n{criteria}\n\nScore:"
        for resp in df["response"]
    ]

    # Load model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # or replace with your own
    device_map = "auto"
    quantization_bits = 4
    temperature = 0.0

    model = get_outlines_model(model_id, device_map, quantization_bits)
    sampler = samplers.greedy()
    generator = generate.json(model, ResponseModel, sampler=sampler)

    # Run generation
    with torch.no_grad():
        results = generator(prompts)

    scores = [r.score for r in results]
    df["score"] = scores

    return df


# Gradio UI
iface = gr.Interface(
    fn=label_responses,
    inputs=[
        gr.Textbox(label="Story", lines=6, placeholder="Enter the story..."),
        gr.Textbox(label="Question", lines=2, placeholder="Enter the question..."),
        gr.Textbox(label="Criteria", lines=4, placeholder="Enter the evaluation criteria..."),
        gr.File(label="Responses CSV (.csv with 'response' column)", file_types=[".csv"]),
    ],
    outputs=gr.Dataframe(label="Labeled Responses", type="pandas"),
    title="Zero-Shot Evaluation",
    description="Evaluate responses to a story-question pair using criteria. Upload a CSV with a 'response' column.",
)

if __name__ == "__main__":
    iface.launch()
