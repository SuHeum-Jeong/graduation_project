# Voice Phishing Detection with Llama 3.1 8B + LoRA

이 프로젝트는 한국어 통화 데이터를 기반으로 다음 3단계를 수행하기 위한 작업 공간입니다.

1. 원본 `Llama 3.1 8B` 모델 성능 평가
2. `LoRA` 파인튜닝
3. 파인튜닝 모델 성능 평가

현재 저장소에는 아래 작업이 먼저 준비되어 있습니다.

- 데이터 디렉토리 구조
- 원본 대화 기준 `train/valid/test` split
- `conversation/prefix/final` JSONL 생성
- 데이터 검증 스크립트
- 원본/LoRA 모델 평가 스크립트
- LoRA 학습 스크립트

## 디렉토리 구조

```text
.
├── configs
│   ├── baseline_eval_config.json
│   └── data_pipeline_config.json
├── data
│   ├── augmented
│   │   └── abnormal
│   ├── final
│   ├── labeled
│   │   ├── abnormal
│   │   └── normal
│   ├── raw
│   │   ├── abnormal
│   │   └── normal
│   └── samples
│       ├── conversation
│       └── prefix
├── docs
│   └── dataset_workflow.md
├── src
│   └── common.py
├── outputs
│   ├── baseline_eval
│   └── finetuned_eval
├── scripts
│   ├── build_labeled_datasets.py
│   ├── evaluate_base_model.py
│   ├── make_split_manifest.py
│   └── split_dataset.py
└── 데이터 전처리.pdf
```

## 데이터 배치 규칙

- 원본 정상 통화: `data/raw/normal/raw_<conversation_id>.json`
- 원본 비정상 통화: `data/raw/abnormal/raw_<conversation_id>.json`
- 증강 비정상 통화: `data/augmented/abnormal/aug_<conversation_id>.json`
- 문장별 라벨링 완료본:
  - 정상: `data/labeled/normal/labeled_<conversation_id>.json`
  - 비정상 원본/증강: `data/labeled/abnormal/labeled_<conversation_id>.json`

## 실행 순서

1. 원본/라벨링/증강 파일을 위 경로에 배치
2. split 및 JSONL 생성

```bash
python3 scripts/make_split_manifest.py --config configs/data_pipeline_config.json
```

3. 데이터 검증

```bash
python3 scripts/validate_dataset.py \
  --train_file data/final/train.jsonl \
  --valid_file data/final/valid.jsonl \
  --test_file data/final/test.jsonl \
  --split_manifest data/splits/split_manifest_v1.json
```

4. 원본 모델 평가

```bash
python3 scripts/evaluate_base_model.py --config configs/baseline_eval_config.json
```

5. LoRA 학습

```bash
python3 scripts/train_lora.py --config configs/lora_train_config.json
```

6. 파인튜닝 모델 평가

```bash
python3 scripts/evaluate_base_model.py --config configs/finetuned_eval_config.json
```

## 참고

- split은 원본 6자리 대화 ID 기준으로 수행됩니다.
- 비정상 증강 데이터는 `train`에 배정된 원본 비정상 대화에 대해서만 포함됩니다.
- `prefix` 샘플은 5개 발화 단위 누적으로 생성되며, 마지막 구간은 발화 수가 5의 배수가 아니어도 포함됩니다.
- 팀원 레포의 장점이던 `공통 프롬프트/라벨 처리`, `검증`, `LoRA 학습`, `어댑터 평가` 흐름을 현재 디렉토리 구조에 맞게 반영했습니다.
- 스크립트 역할은 분리했습니다:
  - `make_split_manifest.py`: 원본 raw 기준 split 정보 생성
  - `build_labeled_datasets.py`: labeled 기준 conversation/prefix/final 생성
