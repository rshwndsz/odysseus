import logging
from typing import Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/7
# https://github.com/huggingface/transformers/blob/c8c8dffbe45ebef0a8dba4a51024e5e5e498596b/src/transformers/trainer.py#L3694
class ImbalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Change weights for an ad-hoc imbalance fix
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]).to(logits.device))
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1).to(logits.device))
        return (loss, outputs) if return_outputs else loss


# Metrics
# https://huggingface.co/docs/evaluate/en/package_reference/main_classes#evaluate.combine
METRICS = evaluate.combine(["accuracy", "precision", "f1", "recall"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        **METRICS.compute(predictions=predictions, references=labels),
        "1_ratio_diff": np.mean(predictions) - np.mean(labels),  # Sanity check metric
    }


def prepare_model_for_lora(model, lora_config=None):
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if lora_config is None:
        # See https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html for optimal choice of parameters
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
        )
    model = get_peft_model(model, lora_config)
    return model


ID2LABEL = {0: "INCORRECT", 1: "CORRECT"}
LABEL2ID = {"INCORRECT": 0, "CORRECT": 1}


def load_model(
    model_id: str,
    quantization_bits: Optional[int] = 8,
    lora_config=None,
    num_labels: int = 2,
    device_map="auto",
):
    if quantization_bits == 4:
        # https://huggingface.co/docs/peft/main/en/developer_guides/quantization
        logger.info(f"Quantizing model to {quantization_bits}-bits")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_bits == 8:
        logger.info(f"Quantizing model to {quantization_bits}-bits")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        logger.warning("No quantization applied. Make sure this is what you want.")
        quantization_config = None

    print(f"Loading model for sequence classification on {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        low_cpu_mem_usage=device_map or quantization_config,
        quantization_config=quantization_config,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        device_map=device_map,
    )

    if lora_config is not None:
        logger.info("Preparing model for LoRA finetuning")
        model = prepare_model_for_lora(model, lora_config)
        model.print_trainable_parameters()

    return model


def apply_padding_token_fix(tokenizer):
    # https://github.com/huggingface/transformers/issues/23808
    if tokenizer.pad_token_id is not None:
        return tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def apply_padding_token_fix_for_model(model, tokenizer):
    if model.config.pad_token_id is not None:
        return model
    tokenizer = apply_padding_token_fix(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model
