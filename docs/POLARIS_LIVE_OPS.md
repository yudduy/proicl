# POLARIS Live Ops Runbooks

No paid run starts from this document. Launches require an explicit user `go`,
`scripts/run_condition.py --preflight-only` passing, a cost cap, a run-plan
cell, and a clean artifact audit.

## Common Gates

1. Write the plan:
   `python scripts/plan_production_runs.py --out runs/production/plan.json`
2. Refresh official dataset locks:
   `python scripts/lock_datasets.py --tracks math500 humaneval_plus`
   and GPQA only after Hugging Face terms/auth are available.
3. Warm or attach trajectory/model caches under ignored runtime paths.
4. Run preflight-only for the exact cell. It must include explicit
   `--estimated-dollar-cost`, `--cost-cap-dollars`,
   `--trajectory-cache`, and `--user-authorized-paid-run`.
5. Run one small real-slice per track before any final run.
6. Audit the produced bundle as `small_real_slice`; only then project final
   cost from observed throughput.
7. Final launch is blocked unless the projection is under cap or a written
   protocol amendment exists.

## ProRL Resource Profiles

Use `configs/prorl_live_resources.json` and `scripts/launch_prorl_recovery.py`
for the standalone ProRL/BroRL audit. The default order is FarmShare L40S for
free shardable work, Flow 4x A100 80GB for capped weekend acceleration, Modal
for Phase 3/debug bursts, and CloudRift RTX 4090 only as fallback.

No tensor parallelism for the 1.5B audit models. Four GPUs means four
independent one-GPU workers.

GEPA reflection uses xAI with `XAI_REFLECTION_MODEL=grok-4.3`, `$30` initial
cap, and `$100` hard cap. Run `scripts/xai_reflection_smoke.py --dry-run`
before any live reflection call and verify the output redacts `XAI_API_KEY`.

## Modal

Modal Volumes are the cache path for model weights, trajectory replay, and
artifact sync-back. Mount the Volume at a fixed path and call `commit()` after
writing artifacts that must be visible outside the container. Reused containers
must call `reload()` before reading data committed by another container.

Keep Volume v1 file counts below the current performance guidance; use v2 for
large cache trees. Do not rely on filesystem `df` for Volume usage.

GPU requests must be explicit. For 7B scoring, start with the cheapest GPU that
passes vLLM V0/HF parity; do not jump to B200 unless the observed bottleneck
requires it. Modal supports explicit GPU strings such as `L40S`, `A100-40GB`,
`A100-80GB`, `H100`, `H100!`, `H200`, and `B200`.

Modal launch order:

1. Volume/cache warmup.
2. Endpoint health check.
3. One cached replay smoke; expected generation calls: zero.
4. One real-row small slice.
5. `modal logs` monitor until terminal state.
6. Volume commit/sync-back.
7. Local artifact audit.

## Mithril / Flow

Use the same run plan and preflight contract. The backend-specific job wrapper
only owns machine allocation and log transport; it does not choose models,
splits, conditions, or cost caps.

Mithril/Flow launch order:

1. Confirm node/GPU health and cache path.
2. Run exact preflight-only command locally.
3. Submit one small real-slice job with the run-plan cell path.
4. Tail logs until artifact bundle is complete.
5. Sync back artifacts and trajectory cache.
6. Run artifact audit locally.
7. Project final cost from observed throughput.

## Resume

Resume from trajectory cache plus artifact directory. Never overwrite a clean
bundle in place; write to a new seed/stage directory or resume missing cache
keys only. Any resumed final run still requires preflight and a fresh audit.

## Backend Policy

vLLM V0 forced-token scoring remains the accepted MCMC candidate. SGLang is
blocked for MCMC scoring until it passes HF/RWS parity on normalized and
unnormalized token log-prob accounting.
