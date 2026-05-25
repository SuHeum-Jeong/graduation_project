# Remote Eval Guide (RTX 2060 Super, Windows)
RTX 2060 Super, Window 환경에서의
QWen 3.5 2B, Mi:dm 2.0 Base 모델의 baseline 테스트를 위한 명령어 모음입니다.

## 1) MacBook에서 프로젝트 전송

MacBook 터미널에서:

```bash
scp -r /Users/jiyelim/Projects/graduation_project gpuuser@192.168.0.20:~/
```

## 2) 윈도우 원격 접속

MacBook 또는 다른 PC에서:

```bash
ssh gpuuser@192.168.0.20
```

접속 후 PowerShell에서 프로젝트로 이동:

```powershell
cd ~/graduation_project
```

## 3) 환경 세팅 (PowerShell, 수동)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers peft datasets accelerate bitsandbytes sentencepiece wandb
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

W&B로 학습 로그를 확인하려면 학습 실행 전에 같은 PowerShell 세션에서 API key를 설정합니다.

```powershell
$env:WANDB_API_KEY="..."
```

## 4) 스모크 실행 (Midm)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate_base_model.py --config configs/midm/baseline_eval_midm20_base_gpu_smoke8.json
```

## 5) 전체 실행 (Qwen + Midm)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/evaluate_base_model.py --config configs/qwen/baseline_eval_config.json
python scripts/evaluate_base_model.py --config configs/midm/baseline_eval_midm20_base_gpu_full892.json
```

## 결과 경로

- `outputs/baseline_eval_qwen35_2b_base`
- `outputs/baseline_eval_midm20_base_gpu_smoke8`
- `outputs/baseline_eval_midm20_base_gpu_full892`

각 폴더에 `metrics.json`, `report.json`, `predictions.jsonl`이 생성됩니다.
