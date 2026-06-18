"""
LEGACY / DEPRECATED — online multi-turn OpenRLHF agent hook.

The supported training path is offline:
  1) `scripts/batch_run_heuristic.py` → journals
  2) `scripts/export_rlhf_data.py` → `data/offline_decisions.jsonl`
  3) `scripts/prepare_sft_data.py` + `scripts/openrlhf/train_sft.sh`
  4) `scripts/prepare_grpo_data.py` + `scripts/openrlhf/train_grpo_singleturn.sh`

This module is kept for reference only; new work should not depend on it.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch

from aide import Experiment
from aide.policy import ExternalPolicy, SearchAction
from aide.rlhf.ctu_dataset import CTUTask, build_aide_inputs, materialize_workspace
from aide.rlhf.evaluator import (
    compute_step_reward,
    compute_terminal_reward,
    extract_baseline,
)

try:
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "openrlhf is required for aide.rlhf.agent_func. "
        "Install with `pip install aideml[rlhf]`."
    ) from exc


def _parse_action_text(action_text: str) -> SearchAction:
    try:
        payload = json.loads(action_text.strip())
    except json.JSONDecodeError:
        return SearchAction(kind="draft", parent_id=None, rationale="invalid-json", is_invalid=True)

    action = payload.get("action", "draft")
    parent_id = payload.get("parent_id")
    rationale = payload.get("rationale", "")
    if action not in {"draft", "debug", "improve"}:
        return SearchAction(kind="draft", parent_id=None, rationale="invalid-action", is_invalid=True)
    return SearchAction(kind=action, parent_id=parent_id, rationale=rationale)


def _serialize_observation(
    task_desc: str,
    journal_summary: list[dict[str, Any]],
    step_idx: int,
    total_steps: int,
    baseline: float,
    maximize: bool,
) -> str:
    obs = {
        "title": f"AIDE search-policy turn {step_idx}/{total_steps}",
        "task": task_desc,
        "baseline_metric": baseline,
        "higher_is_better": maximize,
        "tree": journal_summary,
        "instruction": (
            'Reply with one JSON object: {"action":"draft|debug|improve","parent_id":"<id>|null","rationale":"<=120 chars"}'
        ),
    }
    return json.dumps(obs)


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.step_idx = 0
        self.total_steps = int(os.getenv("AIDE_RLHF_EPISODE_STEPS", "20"))
        self.prev_best_metric: float | None = None
        self.experiment: Experiment | None = None
        self.external_policy: ExternalPolicy | None = None
        self.baseline_metric = 0.0
        self.maximize = True
        self.dense_rewards = os.getenv("AIDE_RLHF_DENSE_REWARDS", "1") == "1"

    async def reset(self, states: dict, **kwargs) -> dict:
        del kwargs
        self.step_idx = 0
        label = states.get("label")
        if isinstance(label, str):
            label = json.loads(label)
        if not isinstance(label, dict):
            raise ValueError("states['label'] must be a CTU task dict.")
        task = CTUTask(**label)

        workspace_root = Path(os.getenv("AIDE_RLHF_WORKDIR", "workspaces/rlhf"))
        task_workdir = workspace_root / task.row_name
        materialize_workspace(task, task_workdir)

        aide_inputs = build_aide_inputs(task)
        self.experiment = Experiment(
            data_dir=str(task_workdir / "input"),
            goal=aide_inputs["goal"],
            eval=aide_inputs["eval"],
        )
        self.external_policy = ExternalPolicy()
        self.experiment.agent.policy = self.external_policy

        self.baseline_metric, self.maximize = extract_baseline(task.info, task.task_type)
        self.prev_best_metric = None

        obs = _serialize_observation(
            task_desc=str(self.experiment.task_desc),
            journal_summary=self.experiment.journal.summary_for_policy(),
            step_idx=self.step_idx,
            total_steps=self.total_steps,
            baseline=self.baseline_metric,
            maximize=self.maximize,
        )
        return {"observation": obs}

    async def step(self, states: dict, **kwargs) -> dict[str, Any]:
        del kwargs
        if self.experiment is None or self.external_policy is None:
            raise RuntimeError("AgentInstance.reset() must be called before step().")

        action = _parse_action_text(states.get("action_text", "{}"))
        self.external_policy.put_action(action)

        self.experiment.agent.step(exec_callback=self.experiment.interpreter.run)
        self.step_idx += 1
        best_node = self.experiment.journal.get_best_node(only_good=True)
        best_metric = best_node.metric.value if best_node and best_node.metric else None

        reward = compute_step_reward(
            journal=self.experiment.journal,
            baseline_metric=self.baseline_metric,
            maximize=self.maximize,
            prev_best=self.prev_best_metric,
            invalid_action=action.is_invalid,
            dense=self.dense_rewards,
        )
        if best_metric is not None:
            self.prev_best_metric = best_metric

        done = self.step_idx >= self.total_steps
        if done:
            reward += compute_terminal_reward(
                journal=self.experiment.journal,
                baseline_metric=self.baseline_metric,
                maximize=self.maximize,
            )
            self.experiment.interpreter.cleanup_session()

        feedback = _serialize_observation(
            task_desc=str(self.experiment.task_desc),
            journal_summary=self.experiment.journal.summary_for_policy(),
            step_idx=self.step_idx,
            total_steps=self.total_steps,
            baseline=self.baseline_metric,
            maximize=self.maximize,
        )

        reward_t = torch.tensor(float(reward), dtype=torch.float32)
        return {
            "rewards": reward_t,
            "scores": reward_t,
            "environment_feedback": feedback,
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": {
                "step": self.step_idx,
                "best_metric": best_metric,
                "invalid_action": action.is_invalid,
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)

