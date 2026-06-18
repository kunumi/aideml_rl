"""
OpenRLHF reward entrypoint for offline GRPO.

Configure with e.g.:
  --train.reward_func_path aide/rlhf/grpo_reward_entrypoint.py

The trainer typically imports `reward_func` from this module.
"""

from __future__ import annotations

from typing import Any

from .grpo_verifier import reward_one


def reward_func(
    prompts: list[str] | None = None,
    responses: list[str] | None = None,
    labels: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Batch reward compatible with common OpenRLHF call shapes."""
    del prompts, kwargs

    if responses is None:
        responses = []
    if labels is None:
        labels = []

    if len(labels) == 1 and isinstance(labels[0], list):
        labels = labels[0]  # type: ignore[assignment]
    if len(responses) == 1 and isinstance(responses[0], list):
        responses = responses[0]  # type: ignore[assignment]

    out: list[float] = []
    for i, resp in enumerate(responses):
        lab = labels[i] if i < len(labels) else {}
        if isinstance(lab, str):
            import json

            lab = json.loads(lab)
        if not isinstance(lab, dict):
            out.append(-1.0)
            continue
        r = resp if isinstance(resp, str) else str(resp)
        out.append(reward_one(r, lab))
    return out
