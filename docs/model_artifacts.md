# Model and Artifact Sharing

이 프로젝트의 Git 저장소에는 코드, 설정, 문서, 작은 샘플만 올린다. 원본 모델, LoRA adapter, 학습 데이터, 평가 산출물은 Git 밖에서 공유한다.

## What Goes in Git

- `product/`: 제품 시제품 런타임 코드, API, 샘플 transcript
- `src/`: 학습, 평가, 제품이 공유하는 유틸리티 코드
- `scripts/`: 데이터 빌드, 학습, 평가, 시각화 스크립트
- `configs/`: 재현 가능한 실행 설정
- `docs/`: 데이터/모델/운영 절차 문서
- `Dockerfile`, `requirements-eval.txt`, `product/requirements.txt`

## What Stays Outside Git

- Hugging Face에서 내려받는 Qwen 원본 모델 cache
- 파인튜닝된 LoRA adapter 및 full checkpoint
- `data/` 아래 원본/가공 데이터
- `outputs/` 아래 학습 checkpoint, 평가 결과, wandb 로그
- 실제 통화 녹음 파일
- `.env`, API key, Hugging Face token, wandb token

## Base Model

원본 모델은 저장소에 다시 올리지 않는다. 코드와 설정은 Hugging Face model id를 직접 참조한다.

```text
Qwen/Qwen3.5-2B-Base
```

팀원은 `transformers` 실행 시 Hugging Face Hub에서 자동으로 받거나, 필요한 경우 Hugging Face CLI로 미리 로그인한다.

```bash
huggingface-cli login
```

## Fine-Tuned LoRA Adapter

현재 협업 방식은 LoRA adapter를 Google Drive에 올리고 팀원이 내려받는 방식으로 충분하다. 다만 파일만 던져두면 나중에 어떤 모델인지 헷갈리기 쉬우므로, Drive 폴더에는 반드시 버전과 기준 정보를 같이 둔다.

권장 Drive 폴더 구조:

```text
qwen35_2b_base_lora_adapter_v1/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── README.md
└── best_checkpoint_summary.json
```

권장 메타데이터:

- base model: `Qwen/Qwen3.5-2B-Base`
- training config: `configs/qwen/lora_train_config.json`
- source checkpoint: `outputs/checkpoints_qwen35_2b_base/adapter-final`
- dataset version: 예: `split_manifest_v1`
- upload date
- checksum: zip 파일 기준 `sha256sum`

팀원은 adapter를 받은 뒤 아래 경로로 압축을 푼다.

```text
outputs/checkpoints_qwen35_2b_base/adapter-final
```

예시:

```bash
mkdir -p outputs/checkpoints_qwen35_2b_base
unzip qwen35_2b_base_lora_adapter_v1.zip -d outputs/checkpoints_qwen35_2b_base
mv outputs/checkpoints_qwen35_2b_base/qwen35_2b_base_lora_adapter_v1 \
  outputs/checkpoints_qwen35_2b_base/adapter-final
```

압축을 푼 뒤 최소한 아래 파일이 있어야 한다.

```bash
test -f outputs/checkpoints_qwen35_2b_base/adapter-final/adapter_config.json
test -f outputs/checkpoints_qwen35_2b_base/adapter-final/adapter_model.safetensors
```

## Alternative

Google Drive는 빠르게 공유하기 좋다. 장기적으로는 private Hugging Face model repository를 쓰는 편이 더 안정적이다. model card, revision, token 기반 접근 권한, 다운로드 자동화가 모두 한 곳에서 관리되기 때문이다.
