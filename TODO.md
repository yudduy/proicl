# POLARIS TODO

Protocol version: POLARIS-v3.1  
Legacy name: CovComp  
Active goal: execute the standalone ProRL/BroRL Recoverable Fraction audit
through a multi-resource live scheduler: FarmShare L40S for free shardable
work, Mithril/Flow A100 for capped weekend acceleration, Modal for Phase 3/debug
bursts, CloudRift as fallback, and xAI for GEPA reflection. Stop at the first
hard gate failure. Maintain Phase 0 external Karan-Du replication before RF
claims, use GPQA-Diamond only through accepted HF auth or an official
`GPQA_DIAMOND_PATH`, and never seed the main RF numerator memory with
ProRL/BroRL traces. Current Codex `/goal` is active for this audit.

Last protocol sync: 2026-05-15.

## Source Of Truth

- `PROPOSAL.md` defines the scientific contract.
- `TODO.md` defines the operational contract.
- `runs/progress.md` records observed run evidence and decisions.
- If these files conflict, no new experiment may start until the conflict is resolved by a dated progress entry.
- Cited primary papers beat local assumptions when they conflict.

## Protocol Drift Guard

- The active protocol is POLARIS-v3.1.
- Any entry below this file that refers to old fixed-alpha Phase 1/Phase 2 is superseded unless re-approved in a new `DECISION` block.
- `scripts/check_protocol_sync.sh` must pass before any phase transition or new expensive run.
- No run may start without a written split, verifier, sample budget, alpha schedule, stop rule, artifact schema, and cost estimate.
- New protocol edits must be mirrored in `runs/progress.md` in the same work session.

## Current Live State

- Active instance/bid: `polaris-a100x4-ckpt-66304a` is cancelled; `polaris-a100x8-ckpt-e8b2d2` is paused/non-running at `$0.10`.
- Completed job: full-164 official RWS MCMC-only `C0c` reproduction for `Qwen/Qwen2.5-7B` HumanEval.
- Final verified status: `164/164` task JSONs saved locally, duplicate task indices `[]`.
- Final scores from local EvalPlus recompute: HumanEval `91/164 = 0.5549`; HumanEval+ `79/164 = 0.4817`.
- RWS Table 1 comparison: HumanEval target for Qwen2.5-7B power sampling is `0.622`; observed full-run score is `-6.7pp`, within the earlier `±7pp` tolerance but weak enough to treat HumanEval+ as a negative-control/stress-test track.
- The job is complete. It does not authorize proceeding into the old Phase 1/Phase 2 plan.
- New active readiness thread: standalone ProRL/BroRL Recoverable Fraction
  audit before scaling POLARIS. This is a mechanism study on public
  checkpoints, not a final POLARIS scale run.
- Execution target: the active contract is now the checked-in multi-resource
  profile surface. FarmShare continues free queued work when available;
  Mithril/Flow A100 is the capped accelerator; Modal is debug/Phase 3 burst;
  CloudRift is fallback only.
- Credential gates cleared locally on 2026-05-15: HF auth succeeds, GPQA-Diamond
  one-row loader probe succeeds, and xAI GEPA reflection live smoke succeeds
  with redacted output. Secrets live only in ignored `.env`.
- Phase 0 exact-RWS/Karan-Du run state on FarmShare: `8/40` CSV cells complete,
  four cells running, remaining cells pending behind `QOSMaxJobsPerUserLimit`.
- Latest observed Flow quote: A100 80GB SXM in `us-central3-a`, 4x instance
  available at `$0.04/hr` total, below the configured cap
  `$0.025/GPU-hr` (`$0.10/hr` for 4 GPUs). This keeps a 100-hour overrun near
  the `$10` Flow spend cap.

### Preemption Recovery SOP

- The run is checkpointed by task JSON files and heartbeat; preemption can only lose at most the currently in-flight task.
- If a worker disappears, resume only by restarting `scripts/rws_mcmc_ckpt_launch_remote.sh` with the same `RUN_DIR` and `WORLD_SIZE` to continue from completed tasks.
- Current resume parameters: `RUN_DIR=runs/rws_official_full164_mcmc_ckpt`, `RWS_MODEL=qwen`, `WORLD_SIZE=8`, `WORKERS_PER_GPU=2`.
- Do not restart uncheckpointed MCMC scripts for this run: `rws_official_full164_mcmc_generate_remote.sh` or `rws_official_full164_mcmc_one_batch_remote.sh`.

## Revised POLARIS Thesis

POLARIS asks which RL-style reasoning gains can be recovered with pure inference and which require weight updates.

The three core mechanisms are:

1. **Prompt-archive spread:** prompts are latent conditions that induce different reasoning-trajectory distributions.
2. **Diversity-preserving sharpening:** power sampling is used under each prompt, but the alpha schedule must preserve diversity, with mixed-alpha sampling as the default next design.
3. **Verifier-gated memory:** strategies are admitted by external verifier outcomes and tracked with reliability estimates, rather than by LLM self-judgment.

ProRL is the key counterpoint. POLARIS should be strongest in Diminished regimes where RL mostly sharpens existing behavior, and weakest in Sustained regimes where prolonged RL expands the reachable reasoning boundary.

## Immediate Do / Do Not

- Do preserve the completed `C0c` artifacts and final local EvalPlus summary.
- Do compare any future rerun against both the saved `C0c` artifacts and RWS Table 1.
- Do update `runs/progress.md` before any new expensive run.
- Do use `configs/prorl_live_resources.json` plus
  `scripts/launch_prorl_recovery.py` to bind any cell to FarmShare, Flow,
  Modal, or CloudRift. Paid/scale-capable runs still require artifact dir,
  cache path, split, seed, model, backend, cost estimate, cost cap, and
  explicit per-launch authorization.
- Do record launch-time UI/market rates in `costs.json`; public catalog pricing
  is advisory and may differ from the clearing or spot price.
- Do treat R5.2 as blocked until an optimized scorer passes HF/RWS parity; SGLang currently fails by `~9.76` under native SGLang and `~9.40` under `--impl transformers`, while HF/RWS batched MCMC itself passes a bounded GPU verifier smoke.
- Do distinguish smoke readiness from production/final-run readiness. `runs/readiness_smoke.tmp/readiness_report.*` proves local interfaces and artifacts, not final scientific fidelity.
- Do not assume the stale remote partial summary is valid; the valid full-run summary has `task_json_count=164`.
- Do not begin old HumanEval+ Phase 1/Phase 2 unchanged.
- Do not use GPQA official rows unless accepted HF auth or an official local
  `GPQA_DIAMOND_PATH` exists. HF auth is currently present locally; do not print
  GPQA rows or answer keys into tracked docs, logs, or chat.
- Do not claim any recoverable-fraction result before the Phase 0 Karan-Du
  external replication gate passes.
- Do not seed the main RF numerator memory with ProRL/BroRL traces; they are
  diagnostic only.

## Active `/goal` Contract — 2026-05-15

Objective: complete the standalone ProRL/BroRL Recoverable Fraction audit end
to end without stopping until either:

1. Phase 0 exact-RWS/Karan-Du replication passes and Phase 1, filtered Phase 2,
   GEPA/xAI rungs, Phase 3 bucket assignment, canonical artifacts, audits, and
   progress notes are complete; or
2. a hard gate fails, a required artifact/input is missing, or paid-run
   authorization/cost-cap evidence is incomplete.

Execution loop:

- Poll FarmShare Phase 0 exact-RWS until all 40 cells are complete, then run
  `scripts/run_prorl_recovery.py aggregate-rws-exact` and check the `0.748
  +-0.02` Karan-Du gate.
- Poll Flow availability before every paid launch. Flow is considered cheap
  when A100 80GB is at or below `configs/prorl_live_resources.json`
  `flow_a100_weekend.max_bid_dollars_per_gpu_hour = 0.025`. Preferred launch
  band is 4x A100 total price `<= $0.10/hr`.
- If Phase 0 passes, run Phase 1 through the resource-profile launcher with
  4 independent single-GPU workers, then aggregate one canonical
  `phase1_results.parquet`.
- Derive the ProRL-only denominator set from Phase 1. If a task has no trained
  checkpoint improvement, record RF as undefined/no denominator and skip its
  Phase 2/3.
- Run Phase 2 rungs in order. Do not start GEPA/xAI or memory rungs until rungs
  1-4 artifacts and costs are clean.
- Run Phase 3 only after rung 7 exists: base logprob audit, memory transplant
  with same-length cross-family control, prompt sufficiency, ordered bucket
  assignment.
- After every phase transition, append dated evidence to `runs/progress.md` and
  run `bash scripts/check_protocol_sync.sh`.

Paid-run caps:

- Flow A100 initial cap: `$10`; per-launch cap must be present in the command.
- xAI GEPA reflection initial cap: `$30`; hard stop before `$100`.
- Modal debug cap: `$25`.
- CloudRift fallback cap: `$25`, RTX 4090 first, only if FarmShare and capped
  Flow block.

Stop rules:

- Stop immediately if Phase 0 exact-RWS misses the Karan-Du gate.
- Stop immediately if `scripts/check_protocol_sync.sh` fails.
- Stop immediately if any required artifact contract file is missing after a
  run cell claims success.
- Stop before any paid launch if artifact dir, cache path, split, seed, model,
  backend, cost estimate, cost cap, or explicit authorization is missing.

## ProRL Recovery Audit Readiness — 2026-05-13 23:08 UTC

Purpose: establish a standalone Recoverable Fraction audit before scaling
POLARIS. The scientific question is how much of the ProRL/BroRL checkpoint gain
can be recovered by frozen-base inference, and what residual looks weight-only.

Tight `/goal` contract:

- Objective: complete the ProRL Recoverable Fraction audit without stopping
  until resource-profile gates, 100-problem main artifacts, and artifact audits
  are complete, or until a hard gate fails.
- Scope: multi-resource scheduler active with explicit per-launch caps.
  FarmShare is free continuation, Flow A100 is capped acceleration, Modal is
  debug/Phase 3 burst, and CloudRift is fallback. Phase 0 Karan-Du replication
  is required before RF claims. GPQA-Diamond remains blocked without HF/auth or
  `GPQA_DIAMOND_PATH`. ProRL/BroRL traces are diagnostic only and never seed
  main RF numerator memory.

Fixed checkpoints:

- Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
  (`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`).
- ProRL v1: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@v1`
  (`b89048893f95246c6b5749b287f0049e6df42ee9`).
- ProRL v2: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@main`
  (`c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0`). Public label is locked as
  `main`; file ETags are recorded in the model registry because `main` and
  `v2` currently resolve through different refs with matching config/weight
  ETags.
- BroRL: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@brorl`
  (`3441fcdf8c6e81a2959e6352ff50122e3c677d72`).

Required gates:

- Phase 0 external replication gate: Karan-Du/RWS on
  `Qwen/Qwen2.5-Math-7B`, MATH500, `alpha=4`, `max_new_tokens=3072`,
  `block_num=16`, `N_MCMC=10`, target accuracy `0.748 +- 0.02`. HF/vLLM
  internal parity is not enough.
- Phase 1 writes one canonical `phase1_results.parquet` for base, ProRL v1,
  ProRL v2, and BroRL with pass@1/pass@16/pass@128, response length, verifier,
  token ids, and backend.
- Phase 2 is planned only on the ProRL-only denominator set derived from
  `phase1_results.parquet`. If a task has no trained-checkpoint improvement,
  skip Phase 2/3 for that task and record `RF undefined/no denominator`.
- Phase 2 writes rung artifacts once for greedy, BoN
  `K={4,16,64,256,1024}`, RWS MCMC, mixed-alpha MCMC, GEPA archive,
  archive+mixed-alpha, and archive+mixed-alpha+verified memory.
- Phase 3 launch script must assert Phase 1 and Phase 2 rung-7 artifacts exist,
  then compute `phase3_input_set.parquet` once. ProRL traces are diagnostic
  only; they must not seed POLARIS memory in the RF numerator.
- Bucket assignment is locked before Phase 2 unblinding:
  `search_limited`, `prompt_conditional`, `memory_conditional`, `weight_only`.
  `high_base_logprob` is top-quartile normalized base logprob within task
  family. Memory transplant requires transplant pass@16 minus control pass@16
  >= 10 percentage points; control is an unrelated similar-length trajectory
  from a different task family.
- Resource profiles are locked in `configs/prorl_live_resources.json`:
  `farmshare_l40_free`, `flow_a100_weekend`, `modal_burst`, and
  `cloudrift_fallback`. Use one persistent HF/model cache per backend; never
  redownload base/ProRL refs per rung. Flow A100 launches are capped at
  `$0.025/GPU-hr`, preferred 4x total price `<= $0.10/hr`, and `$10` initial
  spend; CloudRift starts with RTX 4090 only if FarmShare and Flow block.
- xAI GEPA reflection uses `XAI_BASE_URL=https://api.x.ai/v1`,
  `XAI_REFLECTION_MODEL=grok-4.3`, `$30` initial cap, and `$100` hard cap.
  The key must remain in ignored local secret files or backend secret stores,
  never in artifacts.
- Token caps lock before any 100-problem run: Phase 0 Karan-Du `3072`,
  MATH500 `4096`, GPQA `2048`, Reasoning Gym `8192`. If a smoke has cap-hit
  rate above 5%, double that task cap once, re-run smoke, then lock it before
  main.
- Smoke ladder: resource probe (`nvidia-smi`, CUDA/PyTorch/vLLM imports, HF
  cache write, snapshot download); xAI dry smoke; then 1 problem x base x
  pass@2 for MATH500 and one Reasoning Gym task; then cost projection; then
  Phase 0 ladder `1 problem -> 20 problems -> full 500` only if projected cost
  is acceptable.

## Team Roles and Edit Surfaces

- Protocol steward: `PROPOSAL.md`, `TODO.md`, `runs/progress.md`.
- Environment agent: Flow/Mithril status, SSH, GPU health, dependency evidence, remote sync.
- RWS replication agent: current `C0c` scripts and RWS Table 1 comparison only.
- Future experiment agent: may act only after the post-`C0c` `DECISION` block defines the next bounded run.
- Scoring agent: metrics derived from saved JSONL/CSV/log artifacts only.

## Required Paper Anchors

Read primary sources before implementing or claiming replication:

- ProRL, arXiv:2505.24864
- P2O, arXiv:2603.21877
- Reasoning with Sampling, arXiv:2510.14901
- Scalable Power Sampling, arXiv:2601.21590
- GEPA, arXiv:2507.19457
- EvalPlus, arXiv:2305.01210
- Dynamic Cheatsheet, arXiv:2504.07952
- ACE, arXiv:2510.04618
- Coverage Principle, arXiv:2510.15020
- MAP-Elites, arXiv:1504.04909
- AlphaEvolve, arXiv:2506.13131
- GPQA, arXiv:2311.12022
- HumanEval, arXiv:2107.03374
- GRPO reference paper, arXiv:2402.03300

## Next Protocol Decision After C0c

`runs/progress.md` contains the dated `C0c Completion And Protocol Decision` block. The selected next direction is:

**POLARIS-MATH500 primary:** switch the first composition MVE to MATH500 as the likely Diminished-regime setting. Keep HumanEval+ as a later Sustained-regime negative-control/stress-test track.

The next goal must still specify expected cost, split, verifier, sample budget, alpha schedule, stop rule, and artifact schema before any new expensive run.

## Completion Criteria For The Active Goal

- Proposal files name POLARIS as the project and encode the revised thesis.
- `TODO.md` and `runs/progress.md` block old Phase 1/Phase 2 from starting unchanged.
- The current `C0c` run is completed or explicitly diagnosed.
- `runs/progress.md` contains the post-`C0c` protocol decision.
- No further expensive phase has started without user approval.

## Publishable Readiness Implementation Checkpoint — 2026-05-13 20:23 UTC

Verdict: **local infrastructure-ready for gated end-to-end rehearsal, not yet a
publishable final result**.

Implemented surfaces:

- `polaris_full_verified_memory` is now a first-class condition with archive,
  power sampling, verifier-gated persistent memory, and archive-build
  provenance requirements.
- `scripts/lock_datasets.py` writes split-level locks for `dev`,
  `small_real_slice`, and `final`; MATH and code are locked with research-style
  separation: MATH optimizer-dev uses `DigitalLearningGmbH/MATH-lighteval`
  `train[0:500]` and final uses full MATH500 `test[0:500]`; code
  optimizer-dev uses MBPP+ `0:378` and final uses full HumanEval+ `0:164`.
  GPQA-Diamond remains `pending_auth` until HF auth or local official cache.
- `scripts/build_polaris_archive.py` emits the archive-memory co-construction
  artifact contract in dry-run/freeze mode: `archive.json`, `memory.sqlite`,
  `memory_events.jsonl`, `descriptor_audit.json`,
  `archive_build_manifest.json`, and `rollouts.json`.
- `scripts/plan_production_runs.py` now writes final cells with split id,
  dataset lock id, archive build id, memory build id, model revision, cost cap,
  artifact dir, cache path, and alpha policy.
- Final artifact audit rejects pending/missing locks, fixture rows, missing
  rollout ledgers, missing memory ledgers for memory conditions, and missing
  archive provenance for GEPA/archive conditions.

Verified local gates:

- `.venv-eval/bin/pytest -q` passed with `150 passed`.
- `bash scripts/check_protocol_sync.sh` passed after this checkpoint.
- `scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp` passed
  with no deferred tracks/experiments and zero cache-replay generation calls.
- `scripts/run_condition.py --condition polaris_full_verified_memory
  --preflight-only` passed against a dry-run frozen archive.

Remaining gates before publishable results:

- Configure accepted GPQA-Diamond access (`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`) or
  official `GPQA_DIAMOND_PATH`, then refresh locks without `pending_auth`.
- Run one cached real row per track and condition, then one small real-slice per
  track under explicit cost cap.
- Run the 200-trace descriptor audit, real GEPA/archive-memory construction on
  dev splits, and artifact-only final analysis.
- Start no final run until small-slice artifact audit is clean and projected
  backend cost is under the user-approved cap.

## Production Live/Final Run Readiness Audit — 2026-05-13

Verdict at time written: **proposal-wide smoke-ready, not production/final-run
ready**. The 20:23 UTC checkpoint above supersedes the implementation-gap
status of this audit but not its science-run gates.

Local evidence is strong: `scripts/smoke_polaris_readiness.py` now exercises every `PROPOSAL.md` experiment family with local fixtures, writes mandated artifact bundles, reports `deferred_tracks=[]`, `deferred_experiments=[]`, and proves cache replay with `generation_calls_on_replay=0`. The local test suite has passed with `128 passed`, and one bounded Modal vLLM batch smoke passed.

That is not enough for high-fidelity final results. The production goal is now to close every gap below before any full experiment claim.

### Proposal-to-repo gap table

| Proposal requirement | Current repo status | Missing for production/final runs | Required gate before launch |
|---|---|---|---|
| §4 offline co-construction loop: build memory on hard dev queries, evolve prompts with memory, freeze archive | `run_mapelite` has bounded injected mutation; `MemoryStore` is in-memory; smoke proves the interface | Real GEPA/P2O-style hard-query loop, persistent memory store, archive+memory co-construction driver, hard-sample mining, costed archive rollouts | CPU dry-run on fixtures plus small real-dev dry-run with persisted `archive.json`, `memory.jsonl/sqlite`, `rollouts.json`, and replayable candidates |
| §4 online inference for math/code/GPQA | MATH500 runner is real; HumanEval+ and GPQA have loaders/verifiers and synthetic smokes | Generic production runner across `math500`, `humaneval_plus`, `gpqa_diamond`; GPQA non-oracle selection path in a real run; HumanEval+ run path with EvalPlus payloads | `scripts/run_condition.py --track ... --preflight-only` plus real small-slice run for each track writing all seven artifacts |
| §5.1 three task families | Data modules exist; HumanEval+/GPQA dev splits are `(0,0)` placeholders | Locked real dev/test splits for HumanEval+ and GPQA-Diamond; dataset cache provenance; no network-dependent final runs | Dataset lockfile with row counts, hashes, split ids, and loader tests against cached data |
| §5.2 three direct replication models | `MODEL_REGISTRY` exists; scripts still mostly use global `MODEL_ID` | CLI/model selection for Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B; model-specific prompt/template checks; optional Llama gate | Model matrix dry-run that writes model id/revision in manifest and rejects benchmark/model mismatches |
| §5.3 baselines | Greedy, BoN, single-prompt power, single-best prompt, archive conditions exist for MATH500 | Dynamic Cheatsheet baseline, ACE baseline, GEPA-only baseline, P2O/published-GRPO comparator ledger, HumanEval+/GPQA baseline runners | Baseline registry and smoke/full-run commands for every baseline claimed in `PROPOSAL.md` |
| §5.4 rollout accounting | `RolloutLedger` and break-even helper exist; MATH500 `costs.json` has token/dollar/wall fields | Charge archive construction, inference, verifier, memory admission/distillation, and baseline rollouts into every production `costs.json`/`rollouts.json` | Artifact audit fails if any production run lacks `rollouts.json` and break-even inputs |
| §6.1 archive-size sweep | Local smoke creates k values | Real k={1,4,8,16,32} archive construction, cache reuse, matched-compute runner | Small real-dev k-sweep dry-run; full sweep pre-registration before paid run |
| §6.2 memory composition | Memory smoke retrieves/admits one entry | Persistent memory modes `{off, raw traces, distilled strategies, cross-task transfer}`, eligible-but-not-retrieved logs, false-positive audit | Memory audit fixtures plus real small-slice memory run with `memory_events.jsonl` |
| §6.3 diversity preservation | Fixed, mixed, and decaying alpha schedules are real for MATH500 | Track-generic alpha scheduling, matched-budget policy in all track runners | Runner emits alpha policy and per-sample alpha distribution; tests verify matched B |
| §6.4 joint optimization | Smoke composes readiness rows | Real selection of best §6.1-§6.3 components and locked comparison to GRPO/P2O | `joint_optimization_plan.json` produced from prior artifacts, not hand-written |
| §6.5 factorial interaction | Ridge-logit smoke exists | Production matrix generator for 2x2x2 `{archive, sharpening, memory}` across deployment volume N | `factorial_design.json` plus analysis script over real run artifacts |
| §6.6 descriptor ablation | Heuristic descriptor and smoke rows exist | 200-trace/track descriptor audit, two independent judge paths, kappa/numeric agreement, surface/random/validation-only archive builders | `descriptor_audit.json` and `descriptor_ablation_plan.json` before descriptor claims |
| §6.7 verifier-gating ablation | Gate smoke exists | LLM-judged memory baseline, stronger-verifier post-hoc false-admission audit | False-admission audit script over admitted memory entries |
| §8 verifier-gated memory details | Basic admission/retrieval/reliability implemented | Track/verifier metadata in memory entries, persistent store, pruning lower-bound rule, eligible-but-not-retrieved logging, LLM distiller schema if used | Memory schema/version tests and replayable memory ledger |
| §10 archive construction | `build_mapelite.py` is MATH500-only and seed-grid by default | Real GEPA integration from vendored source, descriptor novelty/cost Pareto selection, archive freeze for all tracks | Archive builder writes full provenance and can rebuild from cached dev trajectories |
| §11-12 success/falsification | Stats helpers exist | Final aggregation scripts for accuracy, CIs, McNemar, oracle coverage, descriptor success, memory success, break-even N, falsification table | `scripts/analyze_results.py` over complete artifacts; fails on missing conditions |
| §13.1 production backend | vLLM correctness smoke passed; cost projection fails current scale gate | Cost-ready MCMC backend or explicit protocol amendment reducing exact settings/scope; production sharding/resume/monitoring | Fresh bounded Modal/Mithril smoke with projected full-run cost under cap, then explicit user `go` |

### Production-readiness work plan

P0 — **Production run graph and audit contract**

- Add a single source-of-truth run graph for every final experiment family: track, model, split, archive, memory mode, alpha policy, baseline, seed, budget, output path.
- Add `scripts/plan_production_runs.py` that writes `runs/<date>-polaris-production/plan.json` and fails if any proposal-required cell is missing a pre-registration anchor, cost cap, artifact schema, or cache path.
- Add an artifact auditor that distinguishes `smoke`, `small_real_slice`, and `final` runs. Final runs must reject fixture-only evidence.

P1 — **Generic production runner**

- Replace MATH500-only orchestration with `scripts/run_condition.py --track {math500,humaneval_plus,gpqa_diamond} --model <registry-key> --condition <condition>`.
- Keep `scripts/run_math500.py` as a compatibility shim only.
- Production runner must write the seven mandated artifacts plus `rollouts.json`, and must preserve cache keys across restarts.
- GPQA production runner must use `select_gpqa_non_oracle` for selection and `score_gpqa_oracle` only after selection.

P2 — **Dataset/model locks**

- Use optimizer-dev pools outside the final benchmark: MATH train-style rows for
  MATH500, MBPP+/APPS/LiveCodeBench-style rows for HumanEval+, and GPQA
  non-Diamond rows for GPQA-Diamond. Final reporting uses full MATH500,
  full HumanEval+, and full GPQA-Diamond.
- Cache dataset rows locally with row counts and hashes.
- Record HF model revisions for every run. Refuse to inherit `MODEL_ID` from another benchmark.

P3 — **Backend cost/fidelity gate**

- R5 correctness is green, but cost is not. Before Phase 10 or any final run, either:
  1. reduce MH scoring cost with a newly accepted backend path, or
  2. explicitly amend the protocol to a smaller/cheaper high-fidelity run.
- Every backend change must pass HF/RWS parity and a bounded Modal/Mithril smoke before production use.
- SGLang remains blocked for MCMC scoring until parity is fixed.

P4 — **Archive + memory production path**

- Implement real offline co-construction: hard-dev-query mining, GEPA prompt mutation, current-memory context, Pareto/MAP-Elites archive update, frozen archive.
- Persist memory with track/verifier metadata, eligibility logs, retrieval logs, admission/rejection events, reliability posterior snapshots, and pruning decisions.
- Count every archive-construction, verifier, memory-admission, and distillation rollout.

P5 — **Descriptor and ablation production path**

- Implement descriptor reliability audit before descriptor claims: 200 traces/track, two independent judges, categorical agreement, numeric agreement CIs.
- Implement real archive-size, descriptor, memory, diversity, joint-optimization, verifier-gating, and factorial runners from cached trajectories where possible.

P6 — **Baselines and final analysis**

- Add baseline runners/ledgers for Dynamic Cheatsheet, ACE, GEPA-only, P2O/published-GRPO comparison, and the already implemented inference baselines.
- Add final analysis scripts that consume only artifact paths and produce the success/falsification table.
- No result is final unless it is reproducible from `manifest.json`, `candidates.jsonl`, `scores.jsonl`, `costs.json`, `rollouts.json`, and `metrics.json`.

P7 — **Live operations**

- Add production launch/runbook commands for Modal and, if used, Mithril/Flow: cache warmup, preflight, health check, shard/resume, monitor, sync-back, audit.
- Treat `flow bid list` or Modal function submission as insufficient; readiness requires live endpoint/GPU/model checks and artifact writes.

### Production launch rule

No full/final run may start until P0-P3 pass locally and one small real-slice run per target track writes production artifacts. P4-P6 must pass before claiming the full POLARIS thesis. P7 must pass before any long paid launch.

## Superseded POLARIS-MATH500 v1 Pre-Registration

Status: superseded on 2026-05-13 22:28 UTC by the research-faithful split
policy above. Keep this block only as historical context for the earlier
MATH500 slice plan.

- Run directory: `runs/2026-05-12-polaris-math500-v1/`
- Model: `Qwen/Qwen2.5-Math-7B` (frozen). RWS Table 1 (arXiv:2510.14901 §5.1) reports Power-Sampling MATH500 `0.748` on this model — strongest result among the three RWS-tested base families. Distinct from C0c's `Qwen/Qwen2.5-7B`, which was the correct pair for HumanEval (RWS Table 1 HumanEval `0.622`).
- Benchmark: MATH500. Superseded slice plan used test `0-75` and dev `75-100`;
  current policy uses external MATH optimizer-dev rows and full MATH500
  `0:500` for final reporting.
- Verifier: `math/sympy-equivalence-v1` (vendored RWS `parse_answer` + `grade_answer`).
- Selector: `argmax_verifier_score_iteration_tiebreak` (proposal §7; first-wins on ties).
- Archive: k=4. Current archive builds must use external optimizer-dev rows,
  e.g. `scripts/build_mapelite.py --iterations 0 --dev-slice 0 500 --seed 17
  --out runs/2026-05-12-polaris-math500-v1/archive.json`.
- Descriptor extractor: `v1-heuristic-2026-05-12` (heuristic, versioned).
- α schedules: `fixed_alpha_4` (alphas=(4.0,)); `mixed_alpha_4_1` (alphas=(4.0, 1.0), parity cycle).
- MCMC hyperparameters: `MCMC_STEPS=10`, `MCMC_BLOCK_NUM=16` (RWS knob; `BLOCK_SIZE=192` derived), `MAX_NEW_TOKENS=3072`, `temperature=1/alpha`.
- Conditions and per-problem budget `B`:

  | Condition | Sampler | Archive subset | B |
  |---|---|---|---|
  | `greedy` | greedy decode | `direct` only | 1 |
  | `bon_temp1` | low-temp T=1 | `direct` only | 8 |
  | `single_prompt_power` | MCMC α=4 | `direct` only | 8 |
  | `single_best_prompt` | MCMC α=4 | top-dev-fitness cell only | 8 |
  | `full_archive_fixed` | MCMC α=4 | k=4 uniform (2/prompt) | 8 |
  | `full_archive_mixed` | MCMC α=4 (parity even) + low-temp α=1 (parity odd) | k=4 uniform (2/prompt) | 8 |

- Seeds: `{17, 71, 1729}`.
- Mandated artifacts per `runs/<date>/<condition>/seed-<n>/`: `manifest.json`, `archive.json`, `candidates.jsonl`, `scores.jsonl`, `costs.json`, `metrics.json`, `audit.md`.
- Memory disabled (`max_retrieved_memory_entries=0`); admission and reliability loops are not part of v1.
- Stop rule: complete all 6 conditions × 3 seeds × 75 problems. Abort and diagnose on `costs.json` exceeding `$25` per condition or 12 h wall-clock per condition.

No hyperparameter, condition, split, seed, verifier, or selector listed above may be changed without a dated amendment entry in this section and a matching entry in `runs/progress.md`.

## POLARIS-v3.1 R5 Infrastructure Contract — 2026-05-13

R5 is the cost-control layer that must land before MATH500 Phase 10. It is not authorization to run Phase 10.

### Backend decision

- Production candidate: **vLLM V0 for optimized MCMC**. Rationale: RWS scoring requires full-vocabulary logits at forced target positions; vLLM V0 exposes per-request logits processors that can force and score those tokens.
- Accepted vLLM configuration: `VLLM_USE_V1=0`, `dtype=float32`, `model_impl=transformers`, `enable_prefix_caching=True`. BF16/native-vLLM remains unaccepted for MH scoring because parity drift is too large.
- Accepted vLLM MCMC implementation: native vLLM sampling for draft/proposal tokens plus a batched forced-token scoring pass for `lp_norm`/`lp_unnorm`. The fused sampling-recorder path is disabled by default and not accepted for scale-up after Modal H100 diverged at token 105 in a 128-token batched decode.
- vLLM native `logprobs` / `prompt_logprobs` is also not accepted for R5 MH scoring: the Modal V0 probe failed in the sampler assertion path before returning usable aligned scores.
- Correctness oracle: **HF/RWS**. Any optimized MCMC scorer must match HF `score_segments` on small parity tests before paid scale-up.
- SGLang status: **not accepted for MCMC scoring**. Native SGLang and `--impl transformers` failed HF parity in bounded Modal tests. SGLang may still be revisited for shared-prefix generation, but it cannot drive MH ratios now.
- Rejected for R5: zero-token vLLM scoring (`SamplingParams(max_tokens=0)` is invalid), TensorRT-LLM, SGLang/vLLM long-run race without cache, and any training stack.

### Implementation order

1. Trajectory cache first: SQLite cache keyed by `(model_id, track, problem_id, prompt_id, sample_idx, alpha, seed)`, storing generation text, token ids, per-token `lp_norm`/`lp_unnorm`, acceptance ratio, token counts, wall time, dollar cost, and nullable verifier result.
2. Optional cache replay in `core/inference.py`: cache inactive by default; callers must pass model/track/problem/seed context to use it.
3. Local readiness smoke: `scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp` must prove the full CPU loop and write `readiness_report.{json,md}` plus mandated artifacts for every currently runnable MATH500 condition.
4. Paid-run preflight gate: `scripts/run_math500.py` must fail closed unless launch arguments include artifact directory, trajectory cache, split, seed, model, backend, estimated cost, cost cap, and explicit user authorization.
5. vLLM serving surface: greedy, batched low-temp, batched `generate_power`, and `score_segments(prefix_ids_batch, target_segments_batch, temperature)`. The MATH500 runner batches cache-miss candidates per problem and preserves stable cache keys/seeds. Default MCMC uses two-pass sample-then-score until a one-pass fused recorder is GPU-accepted.
6. Modal smokes only: `smoke_sglang_greedy`, `smoke_sglang_parity`, `smoke_cache_replay`, `batched_mcmc_smoke`, isolated `scripts/modal_vllm_app.py::smoke_vllm_parity`, isolated `scripts/modal_vllm_app.py::smoke_vllm_power_path`, and isolated `scripts/modal_vllm_app.py::smoke_vllm_batched_power_path`.
7. No Mithril/Flow submit, Phase 10 launch, or bulk trajectory generation until the user gives a fresh explicit command.

### Acceptance gates

- Local CPU: `pytest tests/unit/` green.
- Cache: duplicate-key rejection/overwrite, verifier mark update, prompt iteration, and inference replay without sampler calls.
- SGLang import/client: clean local import without requiring installed SGLang package; request payloads unit-tested through injected transport.
- vLLM import/client: clean local import without requiring installed vLLM package; forced-token scorer unit-tested through injected fake backend.
- Batched vLLM: fake-backend unit tests must prove draft/proposal calls are batched per block/MCMC step, not per chain; runner cache replay must call no generation path on hits.
- Readiness smoke: local command must emit all mandated artifacts, a prompt-to-artifact checklist, executable rows for every `PROPOSAL.md` experiment family, no deferred rows unless a real external blocker is proven, and `generation_calls_on_replay=0`.
- Paid preflight: `scripts/run_math500.py --preflight-only` and Modal GPU entrypoints must pass only with explicit cost/cache/artifact/authorization fields and fail closed otherwise.
- Parity smoke: HF vs optimized backend `score_segments` absolute difference `< 1e-3` on tiny segments before any MCMC generation scale-up.
- Batched GPU smoke: `scripts/modal_vllm_app.py::smoke_vllm_batched_power_path --batch-size 8 --max-new-tokens 512 --block-num 4 --mcmc-steps 1` must pass with aligned token/logprob arrays.
- Cost discipline: every GPU smoke records target cost, uses persistent caches, and stops after bounded validation.

## POLARIS-v3.0 Protocol Bump — 2026-05-12

`PROPOSAL.md` revised to v3.0. Acronym expanded to **POLARIS: Prompt-Organized Library of Archived Reasoning and Inference Strategies**. The conceptual framing, methodology pseudocode, three-track benchmark matrix, evaluation axis (rollouts / break-even N), four-layer ablations, and five risks/caveats are now the source of truth. Mirror this entry in `runs/progress.md` before any new phase starts.

### What changed vs v2.0

- Primary evaluation axis: **total rollouts at matched accuracy** (break-even N), not dollar cost. Dollars/tokens/latency remain in `costs.json` as diagnostics.
- Three task families locked: MATH500 (Diminished, primary), HumanEval+ (Sustained, expected underperform, negative-control), GPQA-Diamond (Diminished/Plateau, offline only).
- Three replication models locked: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-Math-7B`, `deepseek-ai/deepseek-math-7b-base`. Larger-model run optional.
- Co-construction loop (offline GEPA mutation × verifier-gated memory build) is the methodology, not an extension. v1 memory-disabled remains a deliberate isolation slice.
- Mixed-α sampling is a core mechanism; decaying-α is a §6.3 ablation knob.
- §6.5 factorial 2×2×2 over {archive, sharpening, memory} with logit-scale interaction test is locked.
- §9 trace-aligned descriptors require inter-judge reliability audit (200 traces per track) before any descriptor-alignment claim.
- ProRL recovery question is the overarching frame.

### Repo reorganization plan (R0-R7)

The current `src/polaris/` layout services MATH500 v1 only. v3.0 needs three tracks, three models, memory, descriptor audit, GEPA mutation, rollout accounting, factorial stats. Target layout (locked):

```text
src/polaris/
  config.py                         # MODEL_REGISTRY, TRACK_REGISTRY, defaults
  core/
    archive.py                      # FrozenArchive, PromptEntry — keep
    mixed_alpha.py                  # add decaying-α schedule
    descriptor.py                   # 8 pattern labels + 4 numeric features
    mapelite.py                     # wire real GEPA mutation via vendored/gepa
    memory.py                       # NEW: admission, Beta-Bernoulli, retrieve, distill
    inference.py                    # generalize from MATH500-only to track-agnostic
  infra/
    serving/
      __init__.py                   # Sampler Protocol
      hf.py                         # MOVED from infra/model.py — correctness oracle
      sglang.py                     # shared-prefix generation candidate; MCMC scoring blocked
      sglang_logits.py              # NEW: R5 score_segments/parity helpers
      vllm.py                       # optimized MCMC candidate under v3.1
    mcmc.py                         # algorithm — keep; swap backend via Protocol
  evals/
    datasets/
      math500.py                    # keep
      humaneval_plus.py             # NEW
      gpqa_diamond.py               # NEW
    verifiers/
      math.py                       # keep
      code.py                       # NEW — sandboxed unit-test exec
      gpqa.py                       # NEW — offline answer-key + non-oracle selector
  runners/
    condition_runner.py             # generalize math500.py: track + model + condition
    math500.py                      # thin shim during transition
  stats/                            # NEW
    bootstrap.py
    mcnemar.py
    factorial.py
    breakeven.py
  io/
    artifacts.py, manifest.py       # keep
    rollouts.py                     # NEW — RolloutLedger
    metrics.py                      # demoted to aggregate-only
  vendored/
    rws/ gepa/ dc/                  # keep; evalplus brought back under evals/verifiers/code.py
```

### R0 — Drift-protection contract (this section)

- `TODO.md` POLARIS-v3.0 Protocol Bump entry written (this section).
- `runs/progress.md` mirrors with dated entry.
- `_legacy/` deleted (already-labeled legacy, user-authorized).

### R1 — Serving backend split

- Move `src/polaris/infra/model.py` → `src/polaris/infra/serving/hf.py`. No behavior change.
- New `src/polaris/infra/serving/__init__.py` declares the `Sampler` Protocol that `core/inference.py` already implicitly expects (`generate_greedy`, `generate_low_temp`, `generate_power`).
- New `src/polaris/infra/serving/vllm.py` implements the R5 optimized MCMC candidate.
- New `src/polaris/infra/serving/sglang.py` is retained as the shared-prefix generation candidate, with MCMC scoring blocked until parity is fixed.
- Update imports in: `scripts/modal_app.py`, `scripts/run_math500.py`, `scripts/build_mapelite.py`, `tests/smoke/test_math500_dummy.py`.
- Acceptance: `pytest -q tests/unit` still green.

### R2 — Memory scaffolding

- New `src/polaris/core/memory.py` with:
  - `MemoryEntry` (frozen dataclass: id, strategy_text, descriptor, source_query_id, reliability_alpha, reliability_beta, token_count).
  - `MemoryStore` (admission via independent verifier check, retrieval with descriptor/reliability/cost ranking, Beta-Bernoulli posterior update, prune rule).
  - `distill_strategy(trace) -> str` deterministic extractor + leakage screen.
  - Defaults from `PROPOSAL.md §8`: `max_memory_entries_per_archive=256`, `max_retrieved_memory_entries=3`, `max_retrieved_memory_tokens=512`, `Beta(1,1)` prior.
- Acceptance: unit test `tests/unit/test_memory.py` covering admission gate, posterior update, leakage screen rejecting raw answer.

### R3 — Descriptor expansion (§9 alignment)

- Expand `DESCRIPTOR_CATEGORIES` to the 8 labels: `direct_computation`, `algebraic_transformation`, `case_analysis`, `contradiction`, `backward_verification`, `induction`, `search_enumeration`, `mixed_other`.
- Add 4 numeric features: `step_count`, `branch_count`, `verification_density`, `symbol_diversity`. Return both label and feature dict.
- Bump `DESCRIPTOR_EXTRACTOR_VERSION = "v2-heuristic-2026-05-12"`.
- `MATH500_ARCHIVE_V1` retains its 4-cell composition (subset of the 8). MAP-Elites grid widens to 8 cells under v2/v3 runs.
- Acceptance: existing `tests/unit/test_descriptor.py` still passes; new tests cover the 8 labels.

### R4 — Stats module

- Migrate `bootstrap_ci` and `mcnemar` from `src/polaris/io/metrics.py` → `src/polaris/stats/bootstrap.py` and `src/polaris/stats/mcnemar.py` (keep import shims in `io/metrics.py` for backward compat during transition).
- New `src/polaris/stats/factorial.py`: 2×2×2 logit regression with bootstrap CIs clustered by problem (per `PROPOSAL §6.5`). statsmodels optional dep; fall back to pure numpy + scipy if absent.
- New `src/polaris/stats/breakeven.py`: `break_even_n(polaris_archive_rollouts, polaris_per_query_rollouts, grpo_training_rollouts) -> int` per `PROPOSAL §5.4`.
- Acceptance: `tests/unit/test_stats.py` covers bootstrap CI shape, mcnemar exact-binom path, breakeven crossover algebra.

### R5 — vLLM V0 inference stack and trajectory cache

- Implement `TrajectoryCache` first. The cache is engine-independent and must be green locally before GPU smokes.
- Implement `VLLMGenerator` matching the `Sampler` Protocol for:
  - `generate_greedy(prompt_text, max_new_tokens)`
  - `generate_low_temp(prompt_text, temperature, max_new_tokens)`
  - `score_segments(prefix_ids_batch, target_segments_batch, temperature)`
- Implement bounded `generate_power(...)` through vLLM V0 native sampling plus forced-token scoring after `score_segments` passes HF parity.
- Adapt `infra/mcmc.py`: HF remains the direct-logit correctness oracle; production serving adapters must pass parity before driving MCMC.
- Prefix caching: vLLM `enable_prefix_caching=True` is required; Modal/Mithril launches must keep persistent model/cache volumes.
- Acceptance:
  1. CPU import/unit smoke: SGLang/vLLM modules import locally without installed server packages.
  2. Modal `smoke_vllm_parity`: HF vs vLLM scorer max abs diff `<1e-3`, target `<$0.50`.
  3. Modal `smoke_vllm_power_path`: bounded MCMC mechanics smoke with token/logprob counts aligned, target `<$0.50`.
  4. Modal `smoke_cache_replay`: second pass performs zero generation calls.
- Pre-flight before any Modal launch: re-read RWS Table 1 for MATH500 (model = `Qwen/Qwen2.5-Math-7B`, MCMC_STEPS=10, MCMC_BLOCK_NUM=16, MAX_NEW_TOKENS=3072, α=4). Long runs still require a fresh explicit user command.

### R6 — Three-track scaffolding

- `evals/datasets/humaneval_plus.py` loads HumanEval+ via vendored evalplus data; expose `HUMANEVAL_PLUS_TEST_SLICE` and `HUMANEVAL_PLUS_DEV_SLICE`.
- `evals/datasets/gpqa_diamond.py` loads GPQA-Diamond from a vendored or HF dataset; expose `GPQA_DIAMOND_TEST_SLICE`, `GPQA_DIAMOND_DEV_SLICE`.
- `evals/verifiers/code.py` wraps vendored evalplus sandboxed unit-test exec; `VERIFIER_ID = "code/humaneval-plus-v1"`.
- `evals/verifiers/gpqa.py` exposes `score_gpqa_oracle(generation, reference)` (offline only) and `select_gpqa_non_oracle(candidates) -> Candidate` (majority vote with normalized-likelihood tiebreak per `PROPOSAL §4`).
- Local infrastructure smokes use tiny synthetic fixtures; real runs must point loaders at vendored/cache-backed datasets and still pass paid-run preflight.

### R7 — Generic condition runner

- `src/polaris/runners/condition_runner.py` exposes `run_condition(*, track: TrackConfig, condition: str, archive: FrozenArchive, sampler: Sampler, problems: Sequence[Problem], ...)`. `TrackConfig` from `config.TRACK_REGISTRY` carries verifier callable, descriptor function, default archive composer.
- `src/polaris/runners/math500.py` becomes a 3-line shim importing `run_condition` from `condition_runner.py` with the MATH500 `TrackConfig` pre-bound. v1 callers unchanged.
- New `scripts/run_condition.py` replaces benchmark-specific scripts; argparse: `--track {math500,humaneval_plus,gpqa_diamond} --condition ... --model qwen2.5-math-7b ...`.

### R8 — Cleanup

- `_legacy/` deleted in R0.
- Trim `runs/progress.md` to the last 3 dated decision blocks + `## POLARIS-v3.0 Protocol Bump`. Archive the rest as `runs/_archive/progress-pre-2026-05-12.md`.
- `upstream/` retained until v1 GPU ship, then delete per existing plan.

### Phase sequencing under v3.0

| Phase | Anchor | Description |
|---|---|---|
| **v1** | `TODO.md#polaris-math500-v1` | Already pre-registered. MATH500 fixed-archive, no memory, 6 conditions. Awaiting batched vLLM cost-readiness + explicit user command before Phase 10. |
| **v2** | `TODO.md#polaris-math500-v2-descriptor` | After v1: descriptor inter-judge audit on 200 MATH500 traces. |
| **v3** | `TODO.md#polaris-math500-v3-memory` | Memory enabled. §6.2 composition ablation. Independent-check verifier path required. |
| **v4** | `TODO.md#polaris-humaneval-plus-v1` | HumanEval+ track, Sustained-regime negative-control. |
| **v5** | `TODO.md#polaris-gpqa-diamond-v1` | GPQA-Diamond offline track. Archive oracle coverage, no inference-time selection. |
| **v6** | `TODO.md#polaris-factorial-v1` | Full 2×2×2 factorial on strongest model per regime; break-even N vs published GRPO. |

Each phase requires its own pre-registration block (model, split, B, α-schedule, seeds, archive hash, stop rule, rollout cap) **before** any paid GPU launch. Per-launch user consent still required for every Modal/Mithril/Flow run.

### Drift-protection invariants for the reorg itself

- No new abstraction introduced unless explicitly listed above. Stubs that scaffold future work are allowed; speculative interfaces are not.
- Each file move preserves behavior. `pytest -q tests/unit` must pass before and after each R-block.
- Callsite updates happen in the same commit as the file move. Backward-compat shims (e.g. `runners/math500.py` re-exporting from `condition_runner.py`) are only added when an external script depends on the old import path.
- R5 Modal smokes may run only as bounded validation. Long Mithril/Flow runs and v1 Phase 10 remain blocked until a fresh explicit user command.
- v1 GPU launch is on hold pending batched vLLM cost-readiness, cache replay, and a fresh explicit user command. Do not relaunch the K=8 full-config HF smoke; it is the calibration data point already paid for.
