# Dataset Workflow

## 1. Split 기준

- split 단위는 원본 통화 JSON입니다.
- 기본 비율은 `70 / 15 / 15`입니다.
- 재현성을 위해 고정 시드를 사용합니다.
- 현재 구현은 클래스 불균형을 줄이기 위해 `conversation_label` 기준 stratified split을 기본값으로 사용합니다.

## 2. 증강 데이터 포함 규칙

- 증강 비정상 데이터는 `train` split 원본 비정상 대화의 파생본만 사용합니다.
- `valid`, `test`에는 원본 통화만 포함됩니다.

## 3. 샘플 생성 규칙

### Conversation 샘플

- 통화 전체를 입력으로 사용합니다.
- `target_label`은 `conversation_label`과 동일합니다.
- `parent_conversation_id`는 split 기준이 되는 원본 6자리 대화 ID를 사용합니다.
  증강 샘플도 동일한 원본 ID로 매핑합니다.

### Prefix 샘플

- 발화를 앞에서부터 5개씩 누적합니다.
- 예: `0~4`, `0~9`, `0~14`, `0~끝`
- 정상 통화는 모든 prefix의 `target_label=0`
- 비정상 통화는 현재 prefix 안에 `utterance.label == 1`이 처음 등장하기 전까지 `0`, 이후 `1`

## 4. 산출물

- split manifest: `data/splits/split_manifest_v1.json`
- conversation jsonl:
  - `data/samples/conversation/conversation_train_v1.jsonl`
  - `data/samples/conversation/conversation_valid_v1.jsonl`
  - `data/samples/conversation/conversation_test_v1.jsonl`
- prefix jsonl:
  - `data/samples/prefix/prefix_train_v1.jsonl`
  - `data/samples/prefix/prefix_valid_v1.jsonl`
  - `data/samples/prefix/prefix_test_v1.jsonl`
- final jsonl:
  - `data/final/train.jsonl`
  - `data/final/valid.jsonl`
  - `data/final/test.jsonl`
