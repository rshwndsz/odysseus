import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from peft import LoraConfig, TaskType
from spock import SpockBuilder, spock
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    EarlyStoppingCallback,
    TrainingArguments,
)

import wandb
from odysseus.data import load_and_tokenize_dataset, separate_testset_by_story
from odysseus.utils import (
    METRICS,
    ImbalancedTrainer,
    apply_padding_token_fix,
    apply_padding_token_fix_for_model,
    compute_metrics,
    load_model,
)

logger = logging.getLogger(__name__)


@spock
class FinetuneConfig:
    debug_mode: bool = False

    model_id: str
    num_labels: int = 2
    data_dir: str = "./data"
    dataset_dir: str = "./data/dataset"
    data_fraction: float = 1.0
    metric_for_best_model: str = "eval_accuracy"

    dynamic_padding: bool = True
    max_length: int = 8196
    quantization_bits: Optional[int] = 8
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05

    num_epochs: int = 8
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    lr_scheduler_type: str = "reduce_lr_on_plateau"

    logging_steps: int = 2
    eval_steps: int = 16
    save_steps: int = 64
    save_total_limit: int = 4
    early_stopping_patience: int = 25

    save_dir: str = "./results/finetunes"


def main(config: FinetuneConfig):
    time_id = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
    model_name = "_".join(config.model_id.split("/"))

    logger.info("Loading tokenizer")
    tokenizer = apply_padding_token_fix(AutoTokenizer.from_pretrained(config.model_id))

    logger.info("Loading dataset")
    dataset = load_and_tokenize_dataset(
        dataset_dir=config.dataset_dir,
        tokenizer=tokenizer,
        dynamic_padding=config.dynamic_padding,
        max_length=config.max_length,
    )
    # Get indices of testset samples for each story
    id_2_testdsidx = separate_testset_by_story(dataset["test"], data_dir=config.data_dir)

    logger.info("Loading model")
    if not (config.lora_r is None or config.lora_alpha is None or config.lora_dropout is None):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
    else:
        lora_config = None

    model = load_model(
        model_id=config.model_id,
        quantization_bits=config.quantization_bits,
        lora_config=lora_config,
        num_labels=config.num_labels,
    )
    model = apply_padding_token_fix_for_model(model, tokenizer)

    logger.info("Building trainer")
    if config.dynamic_padding:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8)
    else:
        logger.warning("Using DEFAULT collator. Make sure you're data is already padded to the same size.")
        data_collator = DefaultDataCollator()

    _run = wandb.init()

    output_dir = os.path.join(config.save_dir, f"{time_id}__{model_name}__ft")

    trainer = ImbalancedTrainer(
        model=model,
        args=TrainingArguments(
            remove_unused_columns=True,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            weight_decay=config.weight_decay,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            output_dir=output_dir,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=True,
            report_to="wandb",
            run_name=f"{time_id}__{model_name}__ft",
        ),
        train_dataset=dataset["train"].take(int(len(dataset["train"]) * config.data_fraction)),
        eval_dataset=dataset["validation"].take(int(len(dataset["validation"]) * config.data_fraction)),
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()

    test_out = trainer.predict(dataset["test"])  # type: ignore
    predictions = np.argmax(test_out.predictions, axis=-1)

    # Save predictions
    preds_df = pd.DataFrame(predictions, columns=["predictions"])
    preds_df.to_json(os.path.join(output_dir, "test_preds.json"), orient="table")

    # Per-story metrics
    references = dataset["test"].to_pandas()["label"].to_numpy()  # type: ignore
    perstory_metrics = {}
    for story_id, indices in id_2_testdsidx.items():
        story_preds = predictions[indices]
        story_refs = references[indices]
        perstory_metrics[story_id] = METRICS.compute(predictions=story_preds, references=story_refs)
    perstory_metrics = {f"test/perstory_{k}": v for k, v in perstory_metrics.items()}
    wandb.log(perstory_metrics)


if __name__ == "__main__":
    config = SpockBuilder(FinetuneConfig).generate()
    c = config.FinetuneConfig

    if c.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["WANDB_MODE"] = "offline"
    else:
        logging.basicConfig(level=logging.INFO)

    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # Avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.makedirs(c.save_dir, exist_ok=True)

    main(c)
