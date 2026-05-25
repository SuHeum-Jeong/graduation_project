from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from product.backend.io_utils import load_json
from product.backend.pipeline import DEFAULT_CONFIG_PATH, make_default_output_dir, run_pipeline


def create_app() -> Any:
    try:
        from fastapi import FastAPI, File, UploadFile
    except ImportError as exc:
        raise RuntimeError(
            "API 서버를 실행하려면 fastapi/uvicorn이 필요합니다. "
            "product/requirements.txt를 설치한 뒤 다시 실행하세요."
        ) from exc

    app = FastAPI(title="Voice Phishing Detection Prototype")

    @app.post("/detect")
    def detect(
        file: UploadFile = File(...),
        call_id: str | None = None,
        interval_sec: float = 5.0,
        mock_model: bool = False,
    ) -> dict[str, Any]:
        safe_call_id = call_id or Path(file.filename or "uploaded_call").stem
        output_dir = make_default_output_dir(safe_call_id)
        upload_dir = output_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / (file.filename or "call.wav")
        with audio_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        config = load_json(DEFAULT_CONFIG_PATH)
        return run_pipeline(
            audio_file=audio_path,
            transcript_jsonl=None,
            config=config,
            call_id=safe_call_id,
            output_dir=output_dir,
            interval_sec=interval_sec,
            sleep_enabled=False,
            mock_model=mock_model,
        )

    return app


app = create_app()
