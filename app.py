import gradio as gr
import pandas as pd
import torch
import logging
import textwrap

import outlines
from outlines import Generator
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional

from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = textwrap.dedent("""
You are an assistant tasked with grading answers to a mind reading ability test. You will be provided with the following information:

1. A story that was presented to participants as context
2. The question that participants were asked to answer
3. A grading scheme to evaluate the answers (Correct Responses:1, incorrect response:0, Incomplete response:0, Irrelevant:0)
4. Grading examples
5. A participant answer

Your task is to grade each answer according to the grading scheme. For each answer, you should:

1. Carefully read and understand the answer and compare it to the grading criteria
2. Assigning an score 1 or 0 for each answer.
""").strip()

PROMPT_TEMPLATE = textwrap.dedent("""
<Story>
{story}
</Story>

<Question>
{question}
</Question>

<GradingScheme>
{grading_scheme}
</GradingScheme>

<Answer>
{answer}
</Answer>

Score:""").strip()

class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: Literal["0", "1"]

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

    model = outlines.from_transformers(
        model_id,
        model_kwargs={"device_map": device_map, "quantization_config": quantization_config},
        tokenizer_kwargs={"clean_up_tokenization_spaces": True},
    )
    return model

def format_prompt(story: str, question: str, grading_scheme: str, answer: str) -> str:
    prompt = PROMPT_TEMPLATE.format(
        story=story.strip(),
        question=question.strip(),
        grading_scheme=grading_scheme.strip(),
        answer=answer.strip(),
    )
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
    return full_prompt

def label_responses(story, question, criteria, response_file):
    df = pd.read_csv(response_file.name)
    assert "response" in df.columns, "CSV must contain a 'response' column."

    prompts = [
        format_prompt(story, question, criteria, resp)
        for resp in df["response"]
    ]

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # replace if needed
    device_map = "auto"
    quantization_bits = 4
    temperature = 0.0

    model = get_outlines_model(model_id, device_map, quantization_bits)
    generator = Generator(model, temperature=temperature)

    with torch.no_grad():
        results = generator(prompts)
    
    scores = [r.score for r in results]
    df["score"] = scores

    return df

iface = gr.Interface(
    fn=label_responses,
    inputs=[
        gr.Textbox(label="Story", lines=6, placeholder="Enter the story..."),
        gr.Textbox(label="Question", lines=2, placeholder="Enter the question..."),
        gr.Textbox(label="Criteria (Grading Scheme)", lines=4, placeholder="Enter the evaluation criteria..."),
        gr.File(label="Responses CSV (.csv with 'response' column)", file_types=[".csv"]),
    ],
    outputs=gr.Dataframe(label="Labeled Responses", type="pandas"),
    title="Zero-Shot Evaluation Grader",
    description="Upload a CSV with a 'response' column and provide the story, question, and grading criteria. The model will assign 0/1 scores.",
)

if __name__ == "__main__":
    iface.launch()
