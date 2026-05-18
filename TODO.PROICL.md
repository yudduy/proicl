# ProICL TODO

Active experiment: ProICL Fast-Weight Recovery Audit.

## Contract

Measure how much of the ProRL v2 slow-weight gain is recoverable by frozen-base
fast adaptation:

- `mcmc_only`;
- `gepa_only`;
- `gepa_mcmc`;
- `gepa_mcmc_memory`;
- `prorl_v2_greedy` as the slow-weight reference.

This file is the ProICL operational contract. Older POLARIS TODO entries are
infrastructure history for this session.

## Hard Gates

- Do not make final paper-grade RF claims before the exact Karan-Du/RWS MATH500
  replication gate passes. The 20-problem/B=8 Reasoning Gym ProICL signal run
  is allowed before that gate and must be labeled pilot/signal.
- Do not use ProRL or BroRL trajectories in GEPA evolution or RF numerator
  memory.
- Do not run paid or bulk jobs without explicit user authorization, artifact
  directory, cache path, split, seed, backend, cost estimate, and cost cap.
- Do not use GPQA-Diamond answer keys for inference-time selection.

## Locked Primary Graph

Generate with:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/run_proicl.py plan \
  --root runs/proicl \
  --problem-count 100 \
  --rollout-budget 128 \
  --num-shards 4 \
  --out runs/proicl/run_graph.json
```

Build the dry artifact contract for the cross-task Reasoning Gym archive with:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/run_proicl.py write-direct-archives \
  --root runs/proicl

PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/run_proicl.py build-archive \
  --dry-run \
  --out runs/proicl/archives/proicl_cross_task_gepa \
  --dev-split 0 100 \
  --archive-size 16 \
  --max-metric-calls 1000
```

Live GEPA requires a sampler, scorer, reflection LM, explicit cost cap, and a
fresh user command.

## Analysis Contract

The final decomposition must write:

- `proicl_decomposition.json`;
- `proicl_decomposition.md`.

Required fields:

```text
A_base, A_mcmc, A_gepa, A_gepa_mcmc, A_memory, A_prorl_v2
RF_mcmc, RF_gepa, RF_gepa_mcmc, RF_memory
slow_weight_residual, memory_gain, discovery_gain, composition_gain
```

Interpretation rule: high RF and low RF are both publishable. The result is the
decomposition, not a predetermined win condition.
