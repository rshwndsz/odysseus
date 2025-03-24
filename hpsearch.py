import logging
import os
from datetime import datetime

from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    TrainingArguments,
)

import wandb
from odysseus.data import load_and_tokenize_dataset
from odysseus.utils import (
    ImbalancedTrainer,
    apply_padding_token_fix,
    apply_padding_token_fix_for_model,
    compute_metrics,
    load_model,
)

logger = logging.getLogger(__name__)


def main():
    time_id = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")

    # Init run
    run = wandb.init()
    config = wandb.config
    logging.info(config)

    # Load tokenizer, model, dataset
    tokenizer = apply_padding_token_fix(AutoTokenizer.from_pretrained(config.model_id))
    dataset = load_and_tokenize_dataset(
        dataset_dir=config.dataset_dir,
        tokenizer=tokenizer,
        dynamic_padding=config.use_dynamic_padding,
        max_length=config.max_length,
    )

    if all(v is not None for v in (config.lora_r, config.lora_alpha, config.lora_dropout)):
        logging.info((config.lora_r, config.lora_alpha, config.lora_dropout))
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

    if config.use_dynamic_padding:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8)
    else:
        data_collator = DefaultDataCollator()

    # Build trainer
    trainer = ImbalancedTrainer(
        model=model,
        args=TrainingArguments(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            lr_scheduler_type=config.lr_scheduler_type,
            num_train_epochs=config.num_epochs,
            output_dir=os.path.join(config.save_dir, f"{time_id}_{config.model_id}_{run.id}_hpsearch"),
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            logging_strategy="steps",
            logging_steps=2,
            save_strategy="steps",
            save_steps=128,
            save_total_limit=5,
            load_best_model_at_end=True,
        ),
        train_dataset=dataset["train"].take(int(len(dataset["train"]) * config.data_fraction)),
        eval_dataset=dataset["validation"].take(int(len(dataset["validation"]) * config.data_fraction)),
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
