import json
import os
from typing import Optional

import numpy as np
import torch
from spock import SpockBuilder, spock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from odysseus.data import load_and_tokenize_dataset, separate_testset_by_story
from odysseus.utils import METRICS, apply_padding_token_fix, apply_padding_token_fix_for_model, load_model


@spock
class Config:
    data_dir: str
    device: str

    model_id: str
    num_labels: int
    dataset_dir: str
    dynamic_padding: bool
    max_length: int
    quantization_bits: Optional[int]
    batch_size: int
    save_dir: str


def get_perstory_metrics(config: Config):
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    tokenizer = apply_padding_token_fix(AutoTokenizer.from_pretrained(config.model_id))
    ds = load_and_tokenize_dataset(
        dataset_dir=config.dataset_dir,
        tokenizer=tokenizer,
        dynamic_padding=config.dynamic_padding,
        max_length=config.max_length,
    )
    testset = ds["test"]
    id_2_testdsidx = separate_testset_by_story(testset, data_dir=config.data_dir)
    testset = testset.select_columns(["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(
        testset,  # type: ignore
        batch_size=config.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
    )
    model = load_model(
        model_id=config.model_id,
        quantization_bits=config.quantization_bits,
        lora_config=None,
        num_labels=config.num_labels,
        device_map="cuda",
    )
    model = apply_padding_token_fix_for_model(model, tokenizer)
    model.eval()

    predictions = []
    references = []
    with torch.no_grad():
        pbar = tqdm(total=len(testset))
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)

            logits = outputs.logits
            _preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            _refs = batch["labels"].numpy()

            predictions.extend(_preds)
            references.extend(_refs)
            pbar.update(config.batch_size)

    predictions, references = np.array(predictions), np.array(references)

    # Global metrics
    global_metrics = METRICS.compute(predictions=predictions, references=references)
    global_metrics = {f"test/{k}": v for k, v in global_metrics.items()}
    print(global_metrics)

    # Per-story metrics
    perstory_metrics = {}
    for story_id, indices in id_2_testdsidx.items():
        story_preds = predictions[indices]
        story_refs = references[indices]
        perstory_metrics[story_id] = METRICS.compute(predictions=story_preds, references=story_refs)
    perstory_metrics = {f"test/perstory_{k}": v for k, v in perstory_metrics.items()}

    for k in perstory_metrics:
        print(k)
        print(perstory_metrics[k])

    metrics = {**global_metrics, **perstory_metrics}
    return metrics


if __name__ == "__main__":
    config = SpockBuilder(Config).generate()
    c = config.Config

    save_dir = os.path.join(c.save_dir, "__".join(c.model_id.split("/")[-2:]))
    os.makedirs(save_dir, exist_ok=True)

    metrics = get_perstory_metrics(c)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
