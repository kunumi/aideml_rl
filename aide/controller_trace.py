"""Append-only trace of controller (vLLM) requests and responses."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_log_path: Path | None = None
_log_terminal: bool = False


def configure(*, log_path: Path | str | None = None, terminal: bool | None = None) -> None:
    """Enable controller tracing for the current process."""
    global _log_path, _log_terminal
    if log_path is not None:
        _log_path = Path(log_path)
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _write_session_start()
    elif os.getenv("AIDE_CONTROLLER_LOG"):
        raw = os.environ["AIDE_CONTROLLER_LOG"].strip()
        if raw and raw != "0":
            _log_path = Path(raw)
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            _write_session_start()
    if terminal is not None:
        _log_terminal = terminal
    else:
        _log_terminal = os.getenv("AIDE_CONTROLLER_LOG_TERMINAL", "1") not in (
            "0",
            "false",
            "False",
        )


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _append_record(record: dict) -> None:
    if _log_path is None:
        return
    with _log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_session_start() -> None:
    if _log_path is None or _log_path.exists():
        return
    _append_record({"ts": time.time(), "event": "session_start"})


def log_controller_event(event: str, **fields: Any) -> None:
    """Log a non-LLM controller/policy event (e.g. draft phase skip)."""
    record = {"ts": time.time(), "event": event, **fields}
    _append_record(record)
    if _log_terminal:
        extras = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        print(f"[controller] {event}" + (f" {extras}" if extras else ""), flush=True)


def log_controller_call(
    *,
    node_id: str,
    node_stage: str,
    model: str,
    raw_output: str,
    parsed: Any = None,
    parse_error: str | None = None,
    user_input_chars: int | None = None,
) -> None:
    if _log_path is None:
        return

    record = {
        "ts": time.time(),
        "event": "llm_call",
        "node_id": node_id,
        "node_stage": node_stage,
        "model": model,
        "user_input_chars": user_input_chars,
        "raw_output": raw_output,
        "parsed": _serialize(parsed),
        "parse_error": parse_error,
    }
    _append_record(record)

    if _log_terminal:
        parsed_action = None
        parsed_hint = None
        if parsed is not None and is_dataclass(parsed):
            parsed_action = getattr(parsed, "action", None)
            parsed_hint = getattr(parsed, "hint", None)
        hint_preview = ""
        if parsed_hint:
            hint_preview = parsed_hint.replace("\n", " ")[:120]
            if len(parsed_hint) > 120:
                hint_preview += "..."
        line = f"[controller] node={node_id[:8]} stage={node_stage} action={parsed_action or '?'}"
        if hint_preview:
            line += f' hint="{hint_preview}"'
        if parse_error:
            line += f" parse_error={parse_error}"
        print(line, flush=True)
