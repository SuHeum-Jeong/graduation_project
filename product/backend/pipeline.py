from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from product.backend.io_utils import (
    append_jsonl,
    load_json,
    load_transcript_segments,
    write_json,
    write_jsonl,
    write_transcript_segments,
)
from product.backend.model_contract import build_model_input_row
from product.backend.model_runner import build_model_runner
from product.backend.replay import iter_realtime_rows
from product.backend.risk_policy import RiskPolicy
from product.backend.schemas import AlertEvent, ModelPrediction, TranscriptSegment
from product.backend.stt import FasterWhisperTranscriber


DEFAULT_CONFIG_PATH = Path("product/configs/qwen_base.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio-file", type=Path)
    input_group.add_argument("--transcript-jsonl", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--call-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--interval-sec", type=float, default=None)
    parser.add_argument("--sleep", action="store_true")
    parser.add_argument("--mock-model", action="store_true")
    return parser.parse_args()


def make_default_output_dir(call_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("product/runs") / f"{call_id}_{timestamp}"


def resolve_call_id(args: argparse.Namespace) -> str:
    if args.call_id:
        return str(args.call_id)
    input_path = args.audio_file or args.transcript_jsonl
    return input_path.stem


def load_or_transcribe_segments(
    *,
    audio_file: Path | None,
    transcript_jsonl: Path | None,
    stt_config: dict[str, Any],
) -> tuple[list[TranscriptSegment], str]:
    if transcript_jsonl is not None:
        return load_transcript_segments(transcript_jsonl), "transcript_jsonl"

    if audio_file is None:
        raise ValueError("audio_file or transcript_jsonl is required")

    transcriber = FasterWhisperTranscriber.from_config(stt_config)
    return transcriber.transcribe(audio_file), "faster_whisper"


def maybe_sleep_until_tick(
    *,
    sleep_enabled: bool,
    previous_elapsed_sec: float,
    current_elapsed_sec: float,
) -> None:
    if not sleep_enabled:
        return
    time.sleep(max(current_elapsed_sec - previous_elapsed_sec, 0.0))


def print_event(prefix: str, prediction: ModelPrediction, alert: AlertEvent | None) -> None:
    alert_text = f" alert={alert.alert_level}" if alert else ""
    elapsed_text = f"{prediction.elapsed_sec:.1f}s" if prediction.elapsed_sec is not None else "n/a"
    print(
        f"[{prefix}] elapsed={elapsed_text} prediction={prediction.prediction} "
        f"risk={prediction.risk_score:.4f}{alert_text}",
        flush=True,
    )


def run_pipeline(
    *,
    audio_file: Path | None,
    transcript_jsonl: Path | None,
    config: dict[str, Any],
    call_id: str,
    output_dir: Path,
    interval_sec: float,
    sleep_enabled: bool = False,
    mock_model: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments, stt_engine = load_or_transcribe_segments(
        audio_file=audio_file,
        transcript_jsonl=transcript_jsonl,
        stt_config=dict(config.get("stt", {})),
    )
    transcript_path = output_dir / "transcript.jsonl"
    model_inputs_path = output_dir / "model_inputs.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    alerts_path = output_dir / "alerts.jsonl"

    write_transcript_segments(transcript_path, segments)
    write_jsonl(model_inputs_path, [])
    write_jsonl(predictions_path, [])
    write_jsonl(alerts_path, [])

    runner = build_model_runner(dict(config.get("model", {})), mock_model=mock_model)
    policy = RiskPolicy.from_config(dict(config.get("risk_policy", {})))

    realtime_rows = iter_realtime_rows(
        call_id=call_id,
        segments=segments,
        interval_sec=interval_sec,
    )
    alert_events: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    previous_elapsed_sec = 0.0

    for row in realtime_rows:
        elapsed_sec = float(row["elapsed_sec"])
        maybe_sleep_until_tick(
            sleep_enabled=sleep_enabled,
            previous_elapsed_sec=previous_elapsed_sec,
            current_elapsed_sec=elapsed_sec,
        )
        append_jsonl(model_inputs_path, row)
        prediction = runner.predict(row)
        alert = policy.evaluate_realtime(prediction)
        prediction_row = prediction.to_dict()
        prediction_rows.append(prediction_row)
        append_jsonl(predictions_path, prediction_row)
        if alert:
            alert_row = alert.to_dict()
            alert_events.append(alert_row)
            append_jsonl(alerts_path, alert_row)
        print_event("REALTIME", prediction, alert)
        previous_elapsed_sec = elapsed_sec

    final_row = build_model_input_row(
        call_id=call_id,
        segments=segments,
        mode="final",
    )
    final_row["elapsed_sec"] = max((segment.end_sec for segment in segments), default=0.0)
    append_jsonl(model_inputs_path, final_row)
    final_prediction = runner.predict(final_row)
    final_alert = policy.evaluate_final(final_prediction)
    final_prediction_row = final_prediction.to_dict()
    prediction_rows.append(final_prediction_row)
    append_jsonl(predictions_path, final_prediction_row)
    if final_alert:
        final_alert_row = final_alert.to_dict()
        alert_events.append(final_alert_row)
        append_jsonl(alerts_path, final_alert_row)
    print_event("FINAL", final_prediction, final_alert)

    summary = {
        "call_id": call_id,
        "audio_file": str(audio_file) if audio_file else None,
        "input_transcript_jsonl": str(transcript_jsonl) if transcript_jsonl else None,
        "output_dir": str(output_dir),
        "stt_engine": stt_engine,
        "speaker_mode": "none",
        "interval_sec": interval_sec,
        "segment_count": len(segments),
        "realtime_prediction_count": len(realtime_rows),
        "alert_count": len(alert_events),
        "final_prediction": final_prediction_row,
        "final_alert": final_alert.to_dict() if final_alert else None,
        "artifacts": {
            "transcript_jsonl": str(transcript_path),
            "model_inputs_jsonl": str(model_inputs_path),
            "predictions_jsonl": str(predictions_path),
            "alerts_jsonl": str(alerts_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    call_id = resolve_call_id(args)
    output_dir = args.output_dir or make_default_output_dir(call_id)
    interval_sec = (
        float(args.interval_sec)
        if args.interval_sec is not None
        else float(config.get("runtime", {}).get("interval_sec", 5.0))
    )
    summary = run_pipeline(
        audio_file=args.audio_file,
        transcript_jsonl=args.transcript_jsonl,
        config=config,
        call_id=call_id,
        output_dir=output_dir,
        interval_sec=interval_sec,
        sleep_enabled=bool(args.sleep),
        mock_model=bool(args.mock_model),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
