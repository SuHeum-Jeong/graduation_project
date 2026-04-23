#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_row_label, make_prompt


def load_config(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_train_text(row: dict[str, Any]) -> dict[str, str]:
    label = get_row_label(row)
    if label not in {"0", "1"}:
        label = "0"
    prompt = make_prompt(row.get("instruction", ""), row.get("input", ""))
    return {"train_text": f"{prompt}\n{label}"}


def tokenize_fn(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["train_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lora_train_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "datasets/transformers/peft/torch가 설치되어 있지 않습니다. "
            "학습 환경을 먼저 준비하세요."
        ) from exc

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "valid": config["valid_file"],
        },
    )
    dataset = dataset.map(to_train_text)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(config.get("use_4bit", False))
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=config.get("device_map", "auto"),
    )

    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_length = int(config.get("max_length", 1024))
    tokenized_train = dataset["train"].map(
        lambda rows: tokenize_fn(rows, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_valid = dataset["valid"].map(
        lambda rows: tokenize_fn(rows, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["valid"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=float(config.get("epochs", 3)),
        per_device_train_batch_size=int(config.get("batch_size", 2)),
        per_device_eval_batch_size=int(config.get("batch_size", 2)),
        gradient_accumulation_steps=int(config.get("grad_accum", 8)),
        learning_rate=float(config.get("lr", 2e-4)),
        logging_steps=int(config.get("logging_steps", 20)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(config.get("save_total_limit", 2)),
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    final_dir = os.path.join(config["output_dir"], "adapter-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()
