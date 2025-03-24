import logging
import os
from datetime import datetime
from functools import cache
from typing import Any, Literal, Optional

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from outlines import generate, models, samplers
from pydantic import BaseModel, ConfigDict
from spock import SpockBuilder, spock
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

import wandb
from odysseus.data import separate_testset_by_story
from odysseus.utils import METRICS

logger = logging.getLogger(__name__)


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")  # required by openai
    score: Literal["0", "1"]


@cache
def get_outlines_model(
    model_id: str,
    device_map: str = "auto",
    quantization_bits: Optional[int] = 4,
    openai: bool = False,
):
    if openai:
        model = models.openai(model_id)
        return model

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


def label_data_with_outlines(
    prompts: list[str],
    response_model: Any,
    model_id: str,
    is_openai_model: bool,
    device_map: str = "auto",
    quantization_bits: Optional[int] = 4,
    temperature: float = 0.0,
):
    model = get_outlines_model(
        model_id,
        device_map=device_map,
        quantization_bits=quantization_bits,
        openai=is_openai_model,
    )
    sampler = samplers.multinomial(samples=1, temperature=temperature) if temperature != 0 else samplers.greedy()
    generator = generate.json(model, response_model, sampler=sampler)
    responses = generator(prompts)
    return responses


@spock
class ZeroShotConfig:
    debug_mode: bool = False
    # NOTE: :0 is hardcoded here
    # See also: https://github.com/huggingface/peft/issues/1831#issuecomment-2155627636
    device: str = "cuda:0"

    model_id: str
    is_openai_model: bool = False
    num_labels: int = 2
    dataset_dir: str = "./data/dataset"
    data_dir: str = "./data"

    temperature: float = 0.0
    dynamic_padding: bool = True
    max_length: int = 8196
    quantization_bits: Optional[int] = 8
    batch_size: int = 4
    save_dir: str = "./results/zeroshots"

    wandb_entity: Optional[str] = "rshwndsz"
    wandb_project: Optional[str] = "odysseus"
    wandb_token: Optional[str] = None


def main(config: ZeroShotConfig):
    # https://github.com/huggingface/peft/issues/1831#issuecomment-2155627636
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    # Load text dataset
    ds = load_dataset(config.dataset_dir)
    assert isinstance(ds, DatasetDict), "dataset type out of scope"
    testset = ds["test"]

    id_2_testdsidx = separate_testset_by_story(testset, data_dir=config.data_dir)

    if not config.debug_mode:
        time_id = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
        _run = wandb.init(name=f"{config.model_id}-zeroshot-{time_id}")

    predictions = []
    references = []
    with torch.no_grad():
        for i in tqdm(range(0, len(testset), config.batch_size)):
            batch = testset[i : i + config.batch_size]
            prompts = [text for text in batch["text"]]
            responses = label_data_with_outlines(
                prompts=prompts,
                response_model=ResponseModel,
                model_id=config.model_id,
                is_openai_model=config.is_openai_model,
                device_map=config.device,
                quantization_bits=config.quantization_bits,
                temperature=config.temperature,
            )
            _preds = [int(r.score) for r in responses]  # type: ignore
            _refs = [label for label in batch["label"]]  # type: ignore

            predictions.extend(_preds)
            references.extend(_refs)

    predictions, references = np.array(predictions), np.array(references)

    # Global metrics
    global_metrics = METRICS.compute(predictions=predictions, references=references)
    global_metrics = {f"test/{k}": v for k, v in global_metrics.items()}

    if not config.debug_mode:
        wandb.log(global_metrics)
    logger.info(global_metrics)

    # Per-story metrics
    perstory_metrics = {}
    for story_id, indices in id_2_testdsidx.items():
        story_preds = predictions[indices]
        story_refs = references[indices]
        perstory_metrics[story_id] = METRICS.compute(predictions=story_preds, references=story_refs)
    perstory_metrics = {f"test/perstory_{k}": v for k, v in perstory_metrics.items()}

    if not config.debug_mode:
        wandb.log(perstory_metrics)
    for k in perstory_metrics:
        logger.info(f"{k}\n{perstory_metrics[k]}")

    metrics = {**global_metrics, **perstory_metrics}
    return metrics


if __name__ == "__main__":
    config = SpockBuilder(ZeroShotConfig).generate()
    c = config.ZeroShotConfig

    if c.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["WANDB_MODE"] = "offline"
        logging.debug("RUNNING IN DEBUG MODE")
    else:
        logging.basicConfig(level=logging.INFO)
    if c.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = c.wandb_entity
    if c.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = c.wandb_project
    if c.wandb_token is not None:
        os.environ["WANDB_API_KEY"] = c.wandb_token

    # Avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.makedirs(c.save_dir, exist_ok=True)

    main(c)
