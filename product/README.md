# Voice Phishing Detection Product Prototype

이 디렉토리는 학습/평가 코드와 분리된 시제품 런타임을 담는다.

## What It Does

`product`는 다음 4단계를 수행한다.

1. 녹음 파일을 STT로 전사한다.
2. 전사된 통화 내용을 현재 평가 데이터와 같은 JSONL row로 변환한다.
3. 5초마다 현재까지의 통화 내용으로 보이스피싱 위험을 판정하고 경고 이벤트를 기록한다.
4. 통화 종료 시 전체 통화 내용으로 최종 판정하고, 보이스피싱으로 판단되면 최종 경고 이벤트를 기록한다.

## Assumption

1차 시제품은 녹음 파일에서 화자를 안정적으로 분리할 수 없다고 가정한다. 따라서 내부 transcript segment에는 `speaker` 필드를 두지 않는다.

```json
{"start_sec": 0.0, "end_sec": 2.8, "text": "안녕하세요. 서울중앙지검입니다."}
```

모델에는 현재 평가 데이터와 같은 JSONL row 형태를 넘긴다. 다만 `input` 안의 통화 내용은 `A:`/`B:` prefix 없이 시간순 전사 문장만 줄바꿈으로 누적한다.

## Model Contract

실시간 판정:

```json
{
  "sample_id": "prefix_demo-call-001_tick_0003",
  "sample_type": "prefix",
  "parent_conversation_id": "demo-call-001",
  "conversation_label": null,
  "target_label": null,
  "prefix_end_idx": 2,
  "instruction": "현재까지의 통화 내용만 보고 보이스피싱 위험 여부를 판단하라. 위험하면 1, 아니면 0만 출력하라.",
  "input": "시간순 전사 내용",
  "elapsed_sec": 15.0
}
```

통화 종료 최종 판정:

```json
{
  "sample_id": "conversation_demo-call-001",
  "sample_type": "conversation",
  "parent_conversation_id": "demo-call-001",
  "conversation_label": null,
  "target_label": null,
  "prefix_end_idx": null,
  "instruction": "다음 통화가 보이스피싱인지 정상인지 판단하라. 보이스피싱이면 1, 정상이면 0만 출력하라.",
  "input": "전체 전사 내용",
  "elapsed_sec": 132.4
}
```

## Replay Demo

샘플 transcript를 5초 단위 모델 입력 row로 변환한다.

```bash
python3 -m product.backend.replay product/samples/demo_call.jsonl --interval-sec 5
```

`--sleep`을 붙이면 실제 시간 간격에 맞춰 출력한다.

## Run Full Pipeline

제품용 의존성을 설치한다.

```bash
python3 -m pip install -r product/requirements.txt
```

녹음 파일을 입력으로 전체 파이프라인을 실행한다.

```bash
python3 -m product.backend.pipeline \
  --audio-file path/to/call.wav \
  --output-dir product/runs/demo-call
```

기본 설정은 다음을 사용한다.

- STT: `faster-whisper`, `small`, Korean
- detector: `Qwen/Qwen3.5-2B-Base` + `outputs/checkpoints_qwen35_2b_base/adapter-final`
- realtime interval: 5 seconds

원본 Qwen 모델은 Git에 포함하지 않고 Hugging Face Hub에서 받는다. 파인튜닝된 LoRA adapter도 Git에 포함하지 않으며, 공유된 artifact를 아래 경로에 둔 뒤 실행한다.

```text
outputs/checkpoints_qwen35_2b_base/adapter-final
```

모델 artifact 공유/다운로드 기준은 루트의 `docs/model_artifacts.md`를 따른다.

실행 결과는 output dir에 저장된다.

```text
transcript.jsonl      # STT 결과
model_inputs.jsonl    # 모델에 들어간 prefix/final JSONL row
predictions.jsonl     # 매 5초 및 최종 모델 판정
alerts.jsonl          # 경고 이벤트만 별도 기록
summary.json          # 실행 요약
```

현재 환경에 STT/모델 패키지가 없거나 모델 다운로드가 어려운 경우, 이미 만든 transcript로 2~4단계를 검증할 수 있다.

```bash
python3 -m product.backend.pipeline \
  --transcript-jsonl product/samples/demo_call.jsonl \
  --output-dir product/runs/demo-transcript \
  --mock-model
```

`--mock-model`은 smoke test 전용이다. 실제 판정은 기본값인 Qwen runner를 사용한다.

## API Server

FastAPI 서버로 녹음 파일 업로드를 받을 수도 있다.

```bash
uvicorn product.backend.api:app --host 0.0.0.0 --port 8000
```

`POST /detect`에 파일을 업로드하면 같은 파이프라인을 실행하고 `summary.json`에 해당하는 응답을 반환한다.

## Current Limitations

녹음 파일 입력은 이미 끝난 파일을 대상으로 하므로, STT가 먼저 파일 전체를 전사한 뒤 segment timestamp를 이용해 5초 단위 실시간 상황을 재현한다. 실제 전화에서 완전한 실시간 탐지를 하려면 streaming STT 입력으로 `TranscriptSegment`가 들어오도록 바꾸면 된다.

화자 분리를 하지 않기 때문에 모델 입력은 `A:`/`B:` 없이 시간순 문장만 포함한다. 기본 detector는 파인튜닝된 LoRA adapter를 붙인 Qwen 모델이며, `product/configs/qwen_base.json`의 threshold는 데모 결과를 보며 조정해야 한다.
