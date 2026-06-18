#!/usr/bin/env python3
"""
LEGACY / DEPRECATED — smoke test for online `AgentInstance` (multi-turn OpenRLHF).

Use the offline pipeline instead (see `aide/rlhf/offline_extractor.py` and README).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap_path()


def _extract_json_object(text: str) -> str:
    """Strip markdown fences and take the first {...} block."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:500]!r}")
    return m.group(0)


def gpt_search_action(observation_text: str, model: str, temperature: float) -> str:
    from aide.backend import query

    user = (
        "You are the AIDE tree-search policy.\n"
        "Read the observation (JSON below) and reply with exactly one JSON object, no markdown:\n"
        '{"action":"draft|debug|improve","parent_id":null or "<node id from tree>","rationale":"<=120 chars"}\n\n'
        "Rules:\n"
        '- "draft": parent_id must be null.\n'
        '- "debug": parent_id must be the id of a buggy leaf node listed in the tree.\n'
        '- "improve": parent_id must be a non-buggy node id you want to refine.\n\n'
        f"Observation JSON:\n{observation_text}"
    )
    raw = query(
        system_message=(
            "You output only a single minified JSON object for the next search action. "
            "No prose, no code fences."
        ),
        user_message=user,
        model=model,
        temperature=temperature,
    )
    if not isinstance(raw, str):
        raise TypeError(f"Expected string completion, got {type(raw)}")
    return _extract_json_object(raw)


async def run_episode(
    label: dict,
    policy_model: str,
    policy_temp: float,
    episode_steps: int,
    workdir: str | None,
) -> None:
    from aide.rlhf.agent_func import AgentInstance

    os.environ["AIDE_RLHF_EPISODE_STEPS"] = str(episode_steps)
    if workdir:
        os.environ["AIDE_RLHF_WORKDIR"] = workdir

    agent = AgentInstance()
    reset_out = await agent.reset({"label": label})
    obs = reset_out["observation"]
    print("--- reset observation (first 800 chars) ---")
    print(obs[:800] + ("..." if len(obs) > 800 else ""))
    print()

    total_reward = 0.0
    while True:
        action_text = gpt_search_action(obs, policy_model, policy_temp)
        print(f"--- policy action ---\n{action_text}\n")

        step_out = await agent.step({"action_text": action_text})
        r = float(step_out["rewards"].item())
        total_reward += r
        logs = step_out.get("extra_logs", {})
        print(
            f"--- step {logs.get('step')} | reward={r:.4f} | "
            f"best_metric={logs.get('best_metric')} | invalid={logs.get('invalid_action')} "
            f"| done={step_out['done']} ---\n"
        )

        if step_out["done"]:
            break
        obs = step_out["environment_feedback"]

    print(f"Episode total reward (sum of step tensors): {total_reward:.4f}")


def main() -> None:
    try:
        import aide.rlhf.agent_func  # noqa: F401 — ensures OpenRLHF is available
    except ImportError as exc:
        print(
            "Missing dependency: OpenRLHF (required for AgentInstance base class).\n"
            "  pip install openrlhf",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    parser = argparse.ArgumentParser(description="Test RLHF AgentInstance with a GPT policy.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/ctu_datasets_info.csv",
        help="CTU index CSV (first row used unless --task_index set).",
    )
    parser.add_argument("--task_index", type=int, default=0, help="Row index into CTU CSV.")
    parser.add_argument(
        "--episode_steps",
        type=int,
        default=int(os.getenv("AIDE_RLHF_EPISODE_STEPS", "3")),
        help="Env horizon (also sets AIDE_RLHF_EPISODE_STEPS).",
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        default=os.getenv("AIDE_TEST_POLICY_MODEL", "gpt-5.2"),
        help="Model id for search policy (GPT on your Azure endpoint).",
    )
    parser.add_argument("--policy_temp", type=float, default=0.3)
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="If set, overrides AIDE_RLHF_WORKDIR for CTU materialization root.",
    )
    args = parser.parse_args()

    from aide.rlhf.ctu_dataset import load_ctu_index

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    tasks = load_ctu_index(csv_path)
    if not tasks:
        raise RuntimeError("No tasks loaded from CSV.")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(f"task_index {args.task_index} out of range (0..{len(tasks)-1})")

    label = asdict(tasks[args.task_index])
    print(f"Task: {label['row_name']} | type={label['task_type']}")
    asyncio.run(
        run_episode(
            label=label,
            policy_model=args.policy_model,
            policy_temp=args.policy_temp,
            episode_steps=args.episode_steps,
            workdir=args.workdir,
        )
    )


if __name__ == "__main__":
    main()
