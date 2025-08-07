import logging
import textwrap
from typing import Literal, Optional

import gradio as gr
import outlines
import pandas as pd
import torch
from outlines import Generator
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "rshwndsz/ft-hermes-3-llama-3.2-3b"
DEVICE_MAP = "auto"
QUANTIZATION_BITS = None
TEMPERATURE = 0.0


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

    if "longformer" in model_id:
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, **{"device_map": device_map, "quantization_config": quantization_config}
        )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, clean_up_tokenization_spaces=True)

    if "longformer" in model_id:
        return hf_model, hf_tokenizer
    else:
        model = outlines.from_transformers(hf_model, hf_tokenizer)
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


def label_single_response(story, question, criteria, response):
    prompt = format_prompt(story, question, criteria, response)

    if "longformer" in MODEL_ID:
        model, tokenizer = get_outlines_model(MODEL_ID, DEVICE_MAP, QUANTIZATION_BITS)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return str(predicted_class)
    else:
        model = get_outlines_model(MODEL_ID, DEVICE_MAP, QUANTIZATION_BITS)
        generator = Generator(model)
        with torch.no_grad():
            result = generator(prompt)
        return result.score


def label_multi_responses(story, question, criteria, response_file):
    df = pd.read_csv(response_file.name)
    assert "response" in df.columns, "CSV must contain a 'response' column."
    prompts = [format_prompt(story, question, criteria, resp) for resp in df["response"]]

    if "longformer" in MODEL_ID:
        model, tokenizer = get_outlines_model(MODEL_ID, DEVICE_MAP, QUANTIZATION_BITS)
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_classes = torch.argmax(logits, dim=1).tolist()
        scores = [str(cls) for cls in predicted_classes]
    else:
        model = get_outlines_model(MODEL_ID, DEVICE_MAP, QUANTIZATION_BITS)
        generator = Generator(model)
        with torch.no_grad():
            results = generator(prompts)
        scores = [r.score for r in results]

    df["score"] = scores
    return df


single_tab = gr.Interface(
    fn=label_single_response,
    inputs=[
        gr.Textbox(label="Story", lines=6),
        gr.Textbox(label="Question", lines=2),
        gr.Textbox(label="Criteria (Grading Scheme)", lines=4),
        gr.Textbox(label="Single Response", lines=3),
    ],
    outputs=gr.Textbox(label="Score"),
)

multi_tab = gr.Interface(
    fn=label_multi_responses,
    inputs=[
        gr.Textbox(label="Story", lines=6),
        gr.Textbox(label="Question", lines=2),
        gr.Textbox(label="Criteria (Grading Scheme)", lines=4),
        gr.File(label="Responses CSV (.csv with 'response' column)", file_types=[".csv"]),
    ],
    outputs=gr.Dataframe(label="Labeled Responses", type="pandas"),
)

iface = gr.TabbedInterface(
    [single_tab, multi_tab],
    ["Single Response", "Batch (CSV)"],
    title="Zero-Shot Evaluation Grader",
)

if __name__ == "__main__":
    iface.launch()
