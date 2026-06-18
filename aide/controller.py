"""Live hindsight controller for AIDE search."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .backend import query
from .journal import Journal, Node
from .rlhf.hint_prompt import (
    ControllerOutput,
    build_history_summary,
    format_controller_input,
    parse_controller_output,
)
from .rlhf.observation import task_desc_to_string
from .utils.config import SearchConfig

logger = logging.getLogger("aide")


class HintController(Protocol):
    def decide(
        self,
        node: Node,
        task_desc: str | dict,
        journal: Journal,
        search_cfg: SearchConfig,
    ) -> ControllerOutput | None: ...


@dataclass
class LLMController:
    """Query a trained controller model via the existing backend."""

    def decide(
        self,
        node: Node,
        task_desc: str | dict,
        journal: Journal,
        search_cfg: SearchConfig,
    ) -> ControllerOutput | None:
        del journal
        if not search_cfg.controller_model:
            return None

        td_str = task_desc_to_string(task_desc)
        user_input = format_controller_input(
            td_str,
            node,
            history_summary=build_history_summary(node),
            max_code_chars=8000,
            max_output_chars=4000,
        )
        from .rlhf.hint_prompt import HINT_SYSTEM_PROMPT

        try:
            out = query(
                system_message=HINT_SYSTEM_PROMPT,
                user_message=user_input,
                model=search_cfg.controller_model,
                temperature=search_cfg.controller_temp,
            )
        except Exception as exc:
            logger.warning("Controller query failed: %s", exc)
            return None

        if not isinstance(out, str):
            return None

        parsed = parse_controller_output(
            out, max_hint_chars=search_cfg.hint_max_chars
        )
        if parsed is None:
            logger.warning("Could not parse controller output: %s", out[:200])
        return parsed


@dataclass
class RandomHintController:
    """Ablation: random hint from an exported pool; action not used."""

    pool_path: str | Path

    def __post_init__(self) -> None:
        self._hints: list[str] = []
        path = Path(self.pool_path)
        if path.is_file():
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    target = d.get("target")
                    if isinstance(target, str):
                        try:
                            obj = json.loads(target)
                            if isinstance(obj, dict) and obj.get("hint"):
                                self._hints.append(str(obj["hint"]))
                        except json.JSONDecodeError:
                            self._hints.append(target)

    def decide(
        self,
        node: Node,
        task_desc: str | dict,
        journal: Journal,
        search_cfg: SearchConfig,
    ) -> ControllerOutput | None:
        del node, task_desc, journal, search_cfg
        if not self._hints:
            return None
        hint = random.choice(self._hints)
        return ControllerOutput(action="improve", hint=hint, confidence=0.5)


def build_controller(search_cfg: SearchConfig) -> HintController | None:
    kind = search_cfg.controller_kind
    if kind == "none":
        return None
    if kind == "llm":
        return LLMController()
    if kind == "random":
        if not search_cfg.hint_pool_path:
            logger.warning("controller_kind=random but hint_pool_path is unset")
            return None
        return RandomHintController(pool_path=search_cfg.hint_pool_path)
    logger.warning("Unknown controller_kind: %s", kind)
    return None
