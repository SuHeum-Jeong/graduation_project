#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_row_label, make_prompt, trim_prompt_text

SYSTEM_MESSAGE = (
    "너는 보이스피싱 통화 여부를 판별하는 이진 분류기다. "
    "입력된 통화가 보이스피싱이 아니면 0, 보이스피싱이면 1을 출력하라. "
    "반드시 0 또는 1 한 글자만 출력하고 다른 설명은 금지한다."
)


def load_config(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_torch_dtype(torch: Any, dtype_name: str | None, fallback: Any) -> Any:
    if not dtype_name:
        return fallback
    dtype = getattr(torch, str(dtype_name), None)
    return dtype if dtype is not None else fallback


def uses_wandb(report_to: Any) -> bool:
    if isinstance(report_to, str):
        return report_to.lower() == "wandb"
    if isinstance(report_to, list):
        return any(str(value).lower() == "wandb" for value in report_to)
    return False


def resolve_chat_template_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "enable_thinking" in config:
        kwargs["enable_thinking"] = bool(config["enable_thinking"])
    return kwargs


def build_chat_prompt(tokenizer: Any, prompt: str, config: dict[str, Any]) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    chat_template_kwargs = resolve_chat_template_kwargs(config)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def to_train_text(row: dict[str, Any], tokenizer: Any, config: dict[str, Any]) -> dict[str, str]:
    label = get_row_label(row)
    if label not in {"0", "1"}:
        label = "0"
    prompt = make_prompt(row.get("instruction", ""), row.get("input", ""))
    prompt_text = build_chat_prompt(tokenizer, prompt, config)
    max_length = int(config.get("max_length", 1024))
    label_token_reserve = max(8, int(config.get("label_token_reserve", 16)))
    prompt_text, _ = trim_prompt_text(
        tokenizer,
        prompt_text,
        max_prompt_tokens=max(1, max_length - label_token_reserve),
    )
    return {
        "prompt_text": prompt_text,
        "train_text": f"{prompt_text}{label}",
    }


def tokenize_fn(examples, tokenizer, max_length: int):
    tokenized = tokenizer(
        examples["train_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    prompt_tokenized = tokenizer(
        examples["prompt_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels: list[list[int]] = []
    for input_ids, attention_mask, prompt_input_ids in zip(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        prompt_tokenized["input_ids"],
    ):
        row_labels = [-100] * len(input_ids)
        prompt_length = min(len(prompt_input_ids), sum(attention_mask))
        for idx, token_id in enumerate(input_ids):
            if attention_mask[idx] == 1 and idx >= prompt_length:
                row_labels[idx] = token_id
        labels.append(row_labels)

    tokenized["labels"] = labels
    return tokenized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen/lora_train_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    try:
        import torch
        from datasets import Features, Value, load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "datasets/transformers/peft/torch가 설치되어 있지 않습니다. "
            "학습 환경을 먼저 준비하세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_file"],
            "valid": config["valid_file"],
        },
        features=Features(
            {
                "sample_id": Value("string"),
                "sample_type": Value("string"),
                "parent_conversation_id": Value("int64"),
                "conversation_label": Value("int64"),
                "target_label": Value("int64"),
                "prefix_end_idx": Value("int64"),
                "instruction": Value("string"),
                "input": Value("string"),
                "output": Value("string"),
            }
        ),
    )
    dataset = dataset.map(lambda row: to_train_text(row, tokenizer, config))

    use_4bit = bool(config.get("use_4bit", False))
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        default_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        compute_dtype = resolve_torch_dtype(
            torch,
            config.get("bnb_4bit_compute_dtype"),
            default_compute_dtype,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": config.get("device_map", "auto"),
    }
    if "max_memory" in config:
        model_kwargs["max_memory"] = {
            int(device) if isinstance(device, str) and device.isdigit() else device: memory
            for device, memory in config["max_memory"].items()
        }

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_kwargs,
    )

    if use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

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

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    report_to = config.get("report_to", "none")
    wandb_project = config.get("wandb_project")
    if wandb_project and uses_wandb(report_to):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    eval_steps = config.get("eval_steps")
    save_steps = config.get("save_steps")
    eval_strategy = "steps" if isinstance(eval_steps, int) and eval_steps > 0 else "epoch"
    save_strategy = "steps" if isinstance(save_steps, int) and save_steps > 0 else "epoch"

    training_arg_values = {
        "output_dir": config["output_dir"],
        "num_train_epochs": float(config.get("epochs", 3)),
        "per_device_train_batch_size": int(config.get("batch_size", 2)),
        "per_device_eval_batch_size": int(config.get("batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("grad_accum", 8)),
        "learning_rate": float(config.get("lr", 2e-4)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "logging_steps": int(config.get("logging_steps", 20)),
        "save_strategy": save_strategy,
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bf16,
        "fp16": torch.cuda.is_available() and not bf16,
        "report_to": report_to,
        "run_name": config.get("run_name"),
        "load_best_model_at_end": bool(config.get("load_best_model_at_end", True)),
        "metric_for_best_model": config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(config.get("greater_is_better", False)),
        "remove_unused_columns": False,
    }
    if eval_strategy == "steps":
        training_arg_values["eval_steps"] = int(eval_steps)
    if save_strategy == "steps":
        training_arg_values["save_steps"] = int(save_steps)
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_args_signature.parameters:
        training_arg_values["eval_strategy"] = eval_strategy
    else:
        training_arg_values["evaluation_strategy"] = eval_strategy
    training_args = TrainingArguments(**training_arg_values)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    final_dir = os.path.join(config["output_dir"], "adapter-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    best_summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
    }
    with open(os.path.join(final_dir, "best_checkpoint_summary.json"), "w", encoding="utf-8") as file:
        json.dump(best_summary, file, ensure_ascii=False, indent=2)
    print(f"Saved LoRA adapter to: {final_dir}")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
