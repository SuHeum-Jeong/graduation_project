# Voice Phishing Detection with Qwen3.5 2B Base + LoRA

한국어 통화 데이터를 기반으로 보이스피싱 여부를 탐지하는 실험용 프로젝트입니다.

현재 흐름은 다음 3단계입니다.

1. 원본 `Qwen/Qwen3.5-2B-Base` baseline 성능 평가
2. `Qwen/Qwen3.5-2B-Base` LoRA 파인튜닝
3. LoRA adapter를 붙인 파인튜닝 모델 성능 평가

`product/`에는 위 모델을 이용하는 제품 시제품 런타임이 들어 있습니다.

## Repository Layout

```text
.
├── Dockerfile
├── requirements-eval.txt
├── configs
│   ├── data_pipeline_config.json
│   ├── midm
│   │   ├── baseline_eval_midm20_base_gpu_full892.json
│   │   └── baseline_eval_midm20_base_gpu_smoke8.json
│   └── qwen
│       ├── baseline_eval_config.json
│       ├── finetuned_eval_config.json
│       └── lora_train_config.json
├── docs
│   ├── dataset_workflow.md
│   ├── model_artifacts.md
│   └── remote_rtx2060_eval.md
├── product
│   ├── README.md
│   ├── configs
│   │   └── qwen_base.json
│   ├── backend
│   └── samples
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

`data/`, `outputs/`, `checkpoints/`, model weights, cache files, local recordings, and secrets are local artifacts and are ignored by Git.

## Git Collaboration Policy

Git에는 팀원이 리뷰하고 수정해야 하는 텍스트 파일만 올립니다.

올리는 것:

- `product/` 제품 시제품 코드와 작은 샘플 transcript
- `src/`, `scripts/`, `configs/`
- `docs/`, `README.md`
- `Dockerfile`, `requirements-eval.txt`, `product/requirements.txt`

올리지 않는 것:

- Qwen 원본 모델 weight와 Hugging Face cache
- 파인튜닝된 LoRA adapter, merged model, checkpoint
- `data/` 아래 원본/가공 데이터
- `outputs/` 아래 학습/평가 산출물
- 실제 통화 녹음 파일
- `.env`, API key, Hugging Face token, wandb token

모델 artifact 공유 방식은 [docs/model_artifacts.md](docs/model_artifacts.md)에 정리되어 있습니다.

## Team Setup

제품 코드만 smoke test하려면 모델 없이도 실행할 수 있습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r product/requirements.txt
python3 -m product.backend.pipeline \
  --transcript-jsonl product/samples/demo_call.jsonl \
  --output-dir product/runs/demo-transcript \
  --mock-model
```

실제 Qwen + LoRA 판정을 실행하려면 다음 artifact가 필요합니다.

- base model: Hugging Face에서 `Qwen/Qwen3.5-2B-Base` 다운로드
- LoRA adapter: 공유된 Google Drive artifact를 `outputs/checkpoints_qwen35_2b_base/adapter-final`에 배치

기본 제품 설정 파일은 `product/configs/qwen_base.json`이며, adapter 기본 경로도 위 위치를 가리킵니다.

## Model Artifact Sharing

원본 Qwen 모델은 Git에 올리지 않습니다. 팀원들은 Hugging Face Hub에서 직접 받습니다.

LoRA adapter는 현재 단계에서는 Google Drive에 올리고 팀원들이 내려받아도 됩니다. 대신 Drive 파일명과 README에 다음 정보를 같이 기록합니다.

- base model: `Qwen/Qwen3.5-2B-Base`
- adapter version: 예: `qwen35_2b_base_lora_adapter_v1`
- training config: `configs/qwen/lora_train_config.json`
- expected local path: `outputs/checkpoints_qwen35_2b_base/adapter-final`
- zip checksum: `sha256sum`

장기적으로는 private Hugging Face model repository가 더 좋습니다. revision, 접근 권한, model card, 자동 다운로드를 한 곳에서 관리할 수 있기 때문입니다.

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
python3 scripts/evaluate_base_model.py --config configs/qwen/baseline_eval_config.json
```

The default baseline config evaluates `Qwen/Qwen3.5-2B-Base` on `data/final/test.jsonl` and writes results to `outputs/baseline_eval_qwen35_2b_base`.

## Docker Evaluation

The Docker image is intended for GPU evaluation. It uses CUDA 12.8 and PyTorch `2.10.0+cu128`, which supports RTX 50-series `sm_120` GPUs.

Build the image:

```bash
docker build -t voice-phishing-eval .
```

Rebuild the image after changing `requirements-eval.txt`, `Dockerfile`, or Python dependencies so the container includes packages such as `wandb`.

Run baseline evaluation:

```bash
mkdir -p outputs .hf-cache
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e USER="${USER:-user}" \
  -e LOGNAME="${LOGNAME:-${USER:-user}}" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/huggingface/torchinductor \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/configs:/workspace/configs" \
  -v "$PWD/scripts:/workspace/scripts" \
  -v "$PWD/src:/workspace/src" \
  -v "$PWD/.hf-cache:/cache/huggingface" \
  voice-phishing-eval \
  python scripts/evaluate_base_model.py --config configs/qwen/baseline_eval_config.json
```

To run a different evaluation config:

```bash
mkdir -p outputs .hf-cache
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/configs:/workspace/configs" \
  -v "$PWD/scripts:/workspace/scripts" \
  -v "$PWD/src:/workspace/src" \
  -v "$PWD/.hf-cache:/cache/huggingface" \
  voice-phishing-eval \
  python scripts/evaluate_base_model.py --config configs/qwen/finetuned_eval_config.json
```

## LoRA Fine-Tuning

Run LoRA training:

```bash
mkdir -p .hf-cache
HF_HOME="$PWD/.hf-cache" HF_HUB_CACHE="$PWD/.hf-cache/hub" TRANSFORMERS_CACHE="$PWD/.hf-cache" \
  python3 scripts/train_lora.py --config configs/qwen/lora_train_config.json
```

Run LoRA training in Docker:

```bash
mkdir -p outputs .hf-cache
export WANDB_API_KEY="..."
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e WANDB_API_KEY \
  -e WANDB_DIR=/workspace/outputs/wandb \
  -e WANDB_CACHE_DIR=/workspace/outputs/wandb \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/configs:/workspace/configs" \
  -v "$PWD/scripts:/workspace/scripts" \
  -v "$PWD/src:/workspace/src" \
  -v "$PWD/.hf-cache:/cache/huggingface" \
  voice-phishing-eval \
  python scripts/train_lora.py --config configs/qwen/lora_train_config.json
```

The training script reads `data/final/train.jsonl` and `data/final/valid.jsonl`.

`data/final/train.jsonl` contains both:

- `conversation` samples for final detection from the full call
- `prefix` samples for early detection from partial call history

The training script uses the same chat-template prompt format as evaluation and masks prompt tokens so loss is applied only to the final `0` or `1` label token.

The default LoRA config is tuned as a conservative QLoRA starting point for dual RTX 5060 Ti GPUs:

- `use_4bit: true`
- `batch_size: 1`
- `grad_accum: 16`
- `lora_r: 16`
- `lora_alpha: 32`

Install `bitsandbytes` and `datasets` in the training environment before running with `use_4bit: true`.

The final adapter is saved to:

```text
outputs/checkpoints_qwen35_2b_base/adapter-final
```

## Fine-Tuned Evaluation

After training, evaluate the LoRA adapter:

```bash
python3 scripts/evaluate_base_model.py --config configs/qwen/finetuned_eval_config.json
```

The finetuned config loads:

- base model: `Qwen/Qwen3.5-2B-Base`
- adapter: `outputs/checkpoints_qwen35_2b_base/adapter-final`
- test file: `data/final/test.jsonl`
- output dir: `outputs/finetuned_eval_qwen35_2b_base`

## Notes

- `enable_thinking` is set to `false` in Qwen evaluation/training configs so the model directly scores or learns the binary `0`/`1` answer.
- Evaluation uses label scoring rather than free-form generation.
- Docker build context excludes local data, outputs, model weights, and Python environments via `.dockerignore`.
