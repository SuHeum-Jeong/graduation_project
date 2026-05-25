FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv "$VIRTUAL_ENV"

COPY requirements-eval.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install -r requirements-eval.txt

RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY README.md .

CMD ["python", "scripts/evaluate_base_model.py", "--config", "configs/qwen/baseline_eval_config.json"]
