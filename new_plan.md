# Hindsight Hint Controller Implementation Plan

## 1. Goal

The goal is to replace or augment AIDE's current heuristic controller with a learned controller that (1) decides how the search tree should be expanded and (2) generates useful natural-language hints for the code-generation LLM.

Instead of training the controller to directly choose a heuristic action or optimize a sparse RL reward, we train it to produce hindsight guidance extracted from completed AIDE search trees.

The controller should learn to answer:

> Given the current node state, what strategic hint would have helped the AIDE coding LLM reach a better child node sooner?

This is closer to search-trace self-distillation than standard online RL.

---

## 2. Core Idea

A completed AIDE run gives us a search tree. Each node contains code, execution results, analysis, metric values, and parent-child relationships.

For a node `n`, we can inspect its future subtree and identify a better descendant or child. Then we can generate a hint that explains the key insight that separates the current node from the better future node.

At training time, the controller sees only the current node state.

At data-generation time, the teacher can see both:

- the current node, and
- a better future node.

This creates supervised training pairs:

```text
input:  current node state
output: {
  action,
  hindsight_hint
}
```

At inference time:

```text
current AIDE node
      ↓
learned controller
      ↓
(action, hint)
      ↓
AIDE expansion logic
      ↓
AIDE coding LLM
      ↓
new candidate script
```

### Controller Outputs

The learned controller is not only a hint generator. It is a learned search policy.

For each node it should output:

```json
{
  "action": "debug | improve | abandon | branch | retry | submit",
  "hint": "short strategic guidance",
  "confidence": 0.0
}
```

The action determines how the node should be expanded.

The hint guides the coding LLM responsible for generating the next candidate.

This separates:

- search control (what to do next),
- code generation guidance (how to do it).

---

## 3. Why This Direction Instead of Subtree-Reward RL

The earlier RL direction was:

```text
node/action -> reward based on best subtree outcome
```

That can still be useful, but it has several drawbacks:

1. Rewards are sparse and noisy.
2. A long code-generation trajectory may receive credit for many unrelated changes.
3. PPO/GRPO would require expensive online rollouts through AIDE.
4. The controller's action is natural language, so the action space is huge.
5. AIDE logs already contain strong offline supervision.

The hint-distillation approach uses the search tree as a hindsight teacher. This gives dense, interpretable supervision before any expensive RL stage.

Recommended training order:

1. supervised fine-tuning on hindsight hints,
2. preference optimization over good vs bad hints,
3. optional online self-distillation,
4. optional PPO/GRPO only after the controller is already useful.

Importantly, the controller is not only learning hints. It is simultaneously learning the tree-expansion policy currently implemented through AIDE heuristics.

This means the project becomes a form of search-policy distillation:

```text
current node
     ↓
(action, hint)
```

rather than pure hint generation.

---

## 4. Data Source

Primary data source:

```text
data/heuristic_runs/logs/**/journal.json
```

Each `journal.json` should be parsed into a tree of AIDE nodes.

Expected useful fields include:

- node id,
- parent id,
- code/script,
- execution output,
- execution status,
- metric value,
- analysis text,
- plan text,
- step/depth,
- dataset/task metadata.

The current script `scripts/export_rlhf_data.py` calls:

```python
from aide.rlhf.exporter import export_logs_dir
```

and exports offline decision rows. That exporter can either be extended or a new exporter can be added for hint-controller training data.

Recommended new module:

```text
aide/rlhf/hint_exporter.py
```

Recommended new script:

```text
scripts/export_hint_controller_data.py
```

---

## 5. Training Example Format

A single SFT example should look like this.

### Input

```text
You are a strategic controller for AIDE.
Your job is to provide a concise hint that will help the coding LLM improve the current solution.
Do not write full code. Give the most useful next insight.

Task:
{task_description}

Dataset metadata:
{dataset_metadata}

Current node:
- depth: {depth}
- status: {status}
- metric: {metric}

Current code:
```python
{current_code}
```

Execution output:
```text
{execution_output}
```

Current analysis:
```text
{current_analysis}
```

Recent history:
```text
{history_summary}
```

Write one strategic hint for the next code-generation step.
```

### Target

```json
{
  "action": "debug",
  "hint": "The code assumes a playerID column exists. Inspect the schema and infer relational keys dynamically instead of hard-coding player identifiers."
}
```

The target should be short, actionable, and diagnostic.

Good target examples:

```text
The current code assumes a playerID column exists, but the relational tables use internal keys such as __PK__. Inspect the schema first and infer joins from actual key columns instead of hard-coding playerID/personID.
```

```text
The failure is caused by mixing datetime and numeric columns before model fitting. Convert datetime features into numeric components or drop them before passing the dataframe to sklearn.
```

```text
The current branch focuses on model tuning, but the larger improvement came from fixing the relational aggregation. Prioritize constructing stable entity-level features before changing the estimator.
```

Bad target examples:

```text
Improve the model.
```

```text
Try XGBoost.
```

```text
Fix the bug.
```

### Action Labels

Action labels are extracted from future tree behavior.

Examples:

```text
Current node crashes
Best child fixes the error
→ action = debug
```

```text
Current node executes successfully but performance improves in future descendants
→ action = improve
```

```text
Current node produces repeated failures and no descendant reaches a competitive solution
→ action = abandon
```

```text
Several promising directions emerge and exploration is beneficial
→ action = branch
```

The exact action taxonomy can evolve, but the initial version should mirror the existing AIDE heuristic controller actions whenever possible.

---

## 6. Choosing the Future Node

Do not always compare the current node with the final best leaf. That can leak too much information and create noisy hints involving many unrelated changes.

Use local future supervision first.

Recommended target selection hierarchy:

### 6.1 Best Immediate Child

For each node `n`, inspect its children and choose the child with the best subtree outcome.

```text
best_child(n) = argmax_child best_reward_in_subtree(child)
```

This creates a local hint:

```text
current node -> best next direction
```

This is the safest and most realistic default.

### 6.2 Best Descendant Within K Steps

If immediate children are too noisy, choose the best descendant within a small horizon.

Recommended values:

```text
K = 2 or 3
```

This captures short chains such as:

```text
schema error -> inspect schema -> correct join key
```

without using the entire future solution as an oracle.

### 6.3 Best Leaf

Use only as an ablation.

This may produce overly broad hints because the final solution may contain many independent improvements.

---

## 7. Reward and Subtree Statistics

Even though this plan is not primarily RL, we still need subtree statistics to select good future nodes.

For every node, compute:

```python
node.best_subtree_metric
node.best_subtree_node_id
node.best_child_by_subtree_metric
node.delta_to_best_child
node.delta_to_best_subtree
```

For metrics where lower is better, normalize direction before comparison.

Recommended normalized score:

```python
normalized_score = metric if higher_is_better else -metric
```

Also track validity:

```python
node.is_valid = execution_succeeded and metric_is_available
```

If no valid descendant exists, either skip the node or create a negative/example-for-preference data point.

---

## 8. Generating Hindsight Hints

The logs usually contain useful `analysis` fields. Use them whenever possible.

There are three possible strategies.

### 8.1 Direct Analysis Distillation

If the better child or descendant already contains a good analysis, use it as the target hint after cleaning.

Example:

```text
better_node.analysis -> target hint
```

Pros:

- cheap,
- grounded in AIDE's own reasoning,
- no extra teacher model call.

Cons:

- analyses may describe what happened after code execution, not what should be advised before generation.

### 8.2 Teacher-Generated Hindsight Hint

Use a strong LLM offline to generate the target hint from current and future nodes.

Teacher prompt:

```text
You are generating training data for a controller that gives hints to a coding LLM.

The controller will only see the current node at inference time.
You, the teacher, can also see a better future node from the completed search tree.

Write a short hint that would have helped the coding LLM move from the current node toward the better node.
Do not reveal exact final code unless the change is local and diagnostic.
Focus on the key reasoning insight.

Current node code:
```python
{current_code}
```

Current execution output:
```text
{current_execution_output}
```

Current analysis:
```text
{current_analysis}
```

Better future node code:
```python
{future_code}
```

Better future execution output:
```text
{future_execution_output}
```

Better future analysis:
```text
{future_analysis}
```

Metric improvement:
{current_metric} -> {future_metric}

Return only the hint.
```

Pros:

- creates cleaner hints,
- can abstract away code diffs,
- closer to the intended controller behavior.

Cons:

- costs teacher-model tokens,
- may hallucinate if not constrained,
- needs filtering.

### 8.3 Diff-Based Hint Generation

Compute a code diff between current and future nodes, then ask the teacher to explain the reason behind the diff.

This is useful when the future node is a direct child or near descendant.

Pros:

- highly grounded,
- easier to audit.

Cons:

- code diffs can be large,
- some improvements are not obvious from diff alone.

Recommended initial implementation:

1. use best immediate child or K-step descendant,
2. generate teacher hints offline,
3. store both raw teacher hint and source future node id,
4. keep direct future analysis as an auxiliary field.

---

## 9. Exported Dataset Schema

Recommended output format: JSONL.

Path:

```text
data/heuristic_runs/hint_controller/sft.jsonl
```

Each line:

```json
{
  "task_id": "...",
  "run_id": "...",
  "node_id": "...",
  "future_node_id": "...",
  "future_selection": "best_child_by_subtree",
  "depth": 3,
  "current_metric": 0.52,
  "future_metric": 0.71,
  "delta_metric": 0.19,
  "valid": true,
  "input": "...",
  "target": "...",
  "metadata": {
    "logs_dir": "...",
    "journal_path": "...",
    "higher_is_better": true
  }
}
```

Also export a preference dataset:

```text
data/heuristic_runs/hint_controller/preferences.jsonl
```

Each line:

```json
{
  "task_id": "...",
  "run_id": "...",
  "node_id": "...",
  "prompt": "...",
  "chosen": "good hindsight hint",
  "rejected": "bad or lower-value branch hint",
  "chosen_future_node_id": "...",
  "rejected_future_node_id": "...",
  "chosen_metric": 0.71,
  "rejected_metric": 0.43
}
```

---

## 10. Training Plan

### 10.1 Stage A: SFT

Train the controller with standard next-token prediction.

Objective:

```text
current node state -> (action, hindsight hint)
```

The controller should be trained as a multi-task model.

Loss:

```text
L = L_action + λ * L_hint
```

where:

- `L_action` is action classification,
- `L_hint` is next-token prediction for the hint.

Recommended base models:

- small instruction model for fast iteration,
- later a stronger coding-oriented model if needed.

Do not start with PPO or GRPO.

SFT is the correct first step because the completed search trees already provide dense supervision.

### 10.2 Stage B: Preference Optimization

After SFT, train with preference pairs.

Possible methods:

- DPO,
- ORPO,
- SimPO.

Recommended first choice:

```text
DPO
```

Preference construction:

```text
hint from better subtree > hint from worse subtree
```

This teaches the controller to prefer hints associated with stronger downstream outcomes.

### 10.3 Stage C: Online Self-Distillation

Run AIDE with the learned controller.

Collect new journals.

Use the new search trees to generate better hints.

Retrain or continue training the controller.

Loop:

```text
run AIDE + controller
      ↓
collect journals
      ↓
extract hindsight hints
      ↓
SFT/DPO update
      ↓
run again
```

This is the stage most similar to on-policy self-distillation.

### 10.4 Stage D: Optional RL

Only after the controller improves AIDE in offline or small online evaluations should PPO/GRPO be considered.

Reward candidates:

```text
score improvement after one generated child
score improvement after K AIDE steps
nodes saved relative to heuristic baseline
final benchmark score under fixed budget
```

GRPO may be preferable to PPO if training multiple sampled hints per state and comparing group-normalized rewards.

However, this should be treated as an optional final stage, not the starting point.

---

## 11. How to Integrate With AIDE

Add a controller mode:

```text
heuristic
controller_policy
heuristic_plus_controller
```

Recommended first integration:

```text
heuristic_plus_controller
```

The heuristic still chooses the broad operation, but the learned controller adds a hint to the coding LLM prompt.

Example prompt insertion:

```text
Controller hint:
{hint}

Use this hint as strategic guidance. You may ignore it if it conflicts with the execution output or dataset schema.
```

This is safer than fully replacing the heuristic controller immediately.

Later ablations:

1. vanilla AIDE heuristic,
2. heuristic + static hint from logs,
3. heuristic + learned hint controller,
4. learned hint controller only,
5. learned hint controller + value model.

---

## 12. Evaluation Plan

Use fixed budgets.

Metrics:

```text
best metric after N nodes
best metric after N LLM calls
best metric after fixed wall-clock budget
number of nodes until first valid solution
number of nodes until baseline-quality solution
final score under same budget
cost in tokens
```

Recommended plots:

```text
best score vs expanded nodes
best score vs token cost
valid solution rate vs expanded nodes
```

Important comparisons:

```text
AIDE heuristic baseline
AIDE heuristic + learned hints
AIDE heuristic + random/ablated hints
AIDE heuristic + future oracle hints, upper bound only
```

The oracle-hint setting is useful only as an upper bound and should not be presented as a deployable method.

---

## 13. Leakage and Safety Checks

Avoid training targets that reveal impossible information.

Do not let hints contain:

- exact final leaderboard score,
- exact final full solution,
- file paths or artifacts unavailable to the current node,
- future execution outputs copied verbatim unless they are local diagnostic errors,
- many independent future changes bundled into one hint.

Prefer hints that are:

- local,
- diagnostic,
- strategy-level,
- grounded in current error/output,
- actionable for the next generation step.

Recommended filters:

```text
max hint length
min metric improvement
future horizon <= K
remove examples with huge code diffs
remove examples where current and future nodes are unrelated rewrites
```

---

## 14. Implementation Steps

### Step 1: Parse journals into trees

Implement:

```text
aide/rlhf/hint_exporter.py
```

Core functions:

```python
load_journal(path) -> list[Node]
build_tree(nodes) -> dict[node_id, Node]
compute_depths(tree) -> None
compute_subtree_best(tree, higher_is_better) -> None
select_future_node(node, strategy="best_child_by_subtree", horizon=2) -> Node | None
```

### Step 2: Build prompt inputs

Implement:

```python
format_controller_input(node, task_metadata, history_summary) -> str
```

Keep this deterministic and versioned.

Add a prompt version field:

```text
hint_prompt_v1
```

### Step 3: Generate target hints

Implement one of:

```python
clean_analysis_as_hint(future_node) -> str
```

or:

```python
generate_teacher_hint(current_node, future_node) -> str
```

Start with the cheaper analysis-based version if teacher calls are not ready.

### Step 4: Export SFT JSONL

Add script:

```text
scripts/export_hint_controller_data.py
```

Suggested CLI:

```bash
python scripts/export_hint_controller_data.py \
  --logs_dir data/heuristic_runs/logs \
  --out data/heuristic_runs/hint_controller/sft.jsonl \
  --ctu_csv data/ctu_datasets_info.csv \
  --future_strategy best_child_by_subtree \
  --horizon 2 \
  --min_delta 0.0
```

### Step 5: Export preference data

Add preference construction:

```text
chosen = hint from better child/descendant
rejected = hint from worse child/descendant
```

Only include pairs where the metric gap is meaningful.

Suggested threshold:

```text
abs(chosen_metric - rejected_metric) >= min_preference_gap
```

### Step 6: Train SFT model

Train using the exported JSONL.

The target is only the hint text.

Keep validation split by dataset/task, not by node, to avoid leakage across the same run.

### Step 7: Offline evaluation

Before plugging into AIDE, evaluate hints with a judge or reranker.

Possible offline checks:

- Does the hint mention information visible in current node?
- Is it actionable?
- Is it shorter than a max length?
- Does it avoid copying future code?
- Does a judge prefer it over heuristic/no hint?

### Step 8: Online AIDE integration

Add the learned hint into the code-generation prompt.

Start with:

```text
heuristic_plus_hint
```

Do not remove the heuristic policy yet.

### Step 9: Online evaluation

Run matched experiments with identical task budgets and seeds where possible.

Compare:

```text
baseline heuristic AIDE
heuristic + learned hint
heuristic + random hint
heuristic + oracle hint
```

---

## 15. Minimal First Milestone

The first milestone should not involve PPO/GRPO.

Minimal milestone:

1. parse existing `journal.json` files,
2. compute subtree-best descendants,
3. export SFT data with current node input and future-analysis target,
4. manually inspect 100 examples,
5. train a small SFT controller that predicts both actions and hints,
6. insert its hint into AIDE prompts,
7. run a small evaluation on held-out relational datasets.

Success criterion:

```text
heuristic + controller reaches a valid/improved solution in fewer nodes than heuristic alone
```

---

## 16. Research Framing

Possible framing:

> We propose hindsight hint distillation for LLM-driven machine learning agents. Completed search traces are converted into supervised training data by propagating future successful reasoning backward as natural-language guidance. A learned controller predicts both search-expansion decisions and strategic hints for the coding LLM, improving search efficiency without requiring expensive online RL.

Main claim:

```text
Search trees contain more useful supervision than the heuristic policy that generated them.
```

Contribution:

```text
A method for converting AIDE search traces into local hindsight coaching signals.
```

Expected empirical result:

```text
The learned hint controller improves sample efficiency under fixed node/token budgets.
```
