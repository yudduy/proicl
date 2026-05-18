# CloudRift Fallback Runbook

Purpose: use CloudRift only as fallback for the ProRL recoverable-fraction audit
when FarmShare and Mithril/Flow block. CloudRift UI spot prices at launch time
are the source of truth for cost estimates.

## GPU Choice

Fallback order:

1. `rtx4090`: fallback probe and 1.5B recovery target.
2. `v100_sxm3`: fallback only after an FP16 smoke confirms the repo path works.
3. `mi350x`: excluded unless NVIDIA capacity disappears and ROCm port risk is
   explicitly accepted.

Do not choose MI350X just for memory. The current repo path is NVIDIA-first.
Do not use CloudRift while FarmShare or capped Flow A100 can run the same cell
cleanly.

## Environment

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export POLARIS_REPO_DIR=/workspace/polaris
export POLARIS_RUN_ROOT=/workspace/polaris/runs/prorl_recovery
```

Use one persistent model cache. Do not redownload base, ProRL v1, ProRL v2, and
BroRL per rung or shard.

## Cost Estimation

Record the CloudRift UI hourly rate in the launch artifact. Public pricing is
only advisory.

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/estimate_cloudrift_cost.py \
  --gpu rtx4090 \
  --hours 2.5 \
  --hourly-rate 0.25
```

Paid runs still fail closed unless `scripts/run_condition.py` receives artifact
dir, cache path, split, seed, model, backend, estimated cost, cost cap, and
`--user-authorized-paid-run`.

## Fallback Probe Ladder

1. `nvidia-smi`
2. CUDA, PyTorch, Transformers, and vLLM imports
3. HF cache write under `$HF_HOME`
4. Snapshot download for:
   - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
   - `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@v1`
   - `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@main`
   - `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@brorl`
5. Tiny smoke: 1 MATH500 problem x base x pass@2 and 1 Reasoning Gym problem x
   base x pass@2, with full artifact audit.

Stop after the tiny smoke and project Phase 1 cost from observed wall/token
stats. Do not start Phase 1 if projected cost exceeds the approved cap.

## Phase Order

1. Phase 0 Karan-Du: 1 problem -> 20 problems -> full 500 only if cost
   projection is acceptable.
2. Phase 1 denominator: write canonical `phase1_results.parquet`.
3. Derive ProRL-only set from Phase 1.
4. Phase 2 filtered ladder on ProRL-only problems, low-cost rungs first.
5. Phase 3 only after rung 7 exists.

Phase 2 planning must use `--phase1-results`; otherwise the CLI fails closed.
