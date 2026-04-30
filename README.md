# Voice Phishing Detection with Qwen3.5 9B + LoRA

한국어 통화 데이터를 기반으로 보이스피싱 여부를 탐지하는 실험용 프로젝트입니다.

현재 흐름은 다음 3단계입니다.

1. 원본 `Qwen/Qwen3.5-9B` baseline 성능 평가
2. `Qwen/Qwen3.5-9B` LoRA 파인튜닝
3. LoRA adapter를 붙인 파인튜닝 모델 성능 평가

## Repository Layout

```text
.
├── Dockerfile
├── requirements-eval.txt
├── configs
│   ├── baseline_eval_config.json
│   ├── data_pipeline_config.json
│   ├── finetuned_eval_config.json
│   └── lora_train_config.json
├── docs
│   └── dataset_workflow.md
├── scripts
│   ├── build_labeled_datasets.py
│   ├── download_public_drive_tree.py
│   ├── evaluate_base_model.py
│   ├── make_split_manifest.py
│   ├── train_lora.py
│   └── validate_dataset.py
└── src
    ├── __init__.py
    └── common.py
```

`data/`, `outputs/`, `checkpoints/`, model weights, cache files are local artifacts and are ignored by Git.

## Data Layout

Place local data under these paths:

- Raw normal calls: `data/raw/normal/raw_<conversation_id>.json`
- Raw abnormal calls: `data/raw/abnormal/raw_<conversation_id>.json`
- Augmented abnormal calls: `data/augmented/abnormal/aug_<conversation_id>.json`
- Labeled normal calls: `data/labeled/normal/labeled_<conversation_id>.json`
- Labeled abnormal calls: `data/labeled/abnormal/labeled_<conversation_id>.json`

The split unit is the original 6-digit conversation ID. Augmented abnormal samples are included only when their parent raw conversation belongs to the `train` split. `valid` and `test` use original 6-digit conversations only.

## Build Datasets

Create the split manifest:

```bash
python3 scripts/make_split_manifest.py --config configs/data_pipeline_config.json
```

Build conversation, prefix, and final JSONL files:

```bash
python3 scripts/build_labeled_datasets.py --config configs/data_pipeline_config.json
```

This creates:

- `data/splits/split_manifest_v1.json`
- `data/samples/conversation/conversation_{train,valid,test}_v1.jsonl`
- `data/samples/prefix/prefix_{train,valid,test}_v1.jsonl`
- `data/final/train.jsonl`
- `data/final/valid.jsonl`
- `data/final/test.jsonl`

Validate generated datasets:

```bash
python3 scripts/validate_dataset.py \
  --train_file data/final/train.jsonl \
  --valid_file data/final/valid.jsonl \
  --test_file data/final/test.jsonl \
  --split_manifest data/splits/split_manifest_v1.json
```

## Baseline Evaluation

Run baseline evaluation locally:

```bash
python3 scripts/evaluate_base_model.py --config configs/baseline_eval_config.json
```

The default baseline config evaluates `Qwen/Qwen3.5-9B` on `data/final/test.jsonl` and writes results to `outputs/baseline_eval`.

## Docker Evaluation

The Docker image is intended for GPU evaluation. It uses CUDA 12.8 and PyTorch `2.10.0+cu128`, which supports RTX 50-series `sm_120` GPUs.

Build the image:

```bash
docker build -t voice-phishing-eval .
```

Run baseline evaluation:

```bash
mkdir -p outputs
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  voice-phishing-eval
```

To run a different evaluation config:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  voice-phishing-eval \
  python scripts/evaluate_base_model.py --config configs/finetuned_eval_config.json
```

## LoRA Fine-Tuning

Run LoRA training:

```bash
python3 scripts/train_lora.py --config configs/lora_train_config.json
```

The training script reads `data/final/train.jsonl` and `data/final/valid.jsonl`.

`data/final/train.jsonl` contains both:

- `conversation` samples for final detection from the full call
- `prefix` samples for early detection from partial call history

The training script uses the same chat-template prompt format as evaluation and masks prompt tokens so loss is applied only to the final `0` or `1` label token.

The final adapter is saved to:

```text
outputs/checkpoints/adapter-final
```

## Fine-Tuned Evaluation

After training, evaluate the LoRA adapter:

```bash
python3 scripts/evaluate_base_model.py --config configs/finetuned_eval_config.json
```

The finetuned config loads:

- base model: `Qwen/Qwen3.5-9B`
- adapter: `outputs/checkpoints/adapter-final`
- test file: `data/final/test.jsonl`
- output dir: `outputs/finetuned_eval`

## Notes

- `enable_thinking` is set to `false` in Qwen evaluation/training configs so the model directly scores or learns the binary `0`/`1` answer.
- Evaluation uses label scoring rather than free-form generation.
- Docker build context excludes local data, outputs, model weights, and Python environments via `.dockerignore`.
