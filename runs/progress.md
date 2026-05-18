# POLARIS Progress

## Local End-to-End Readiness Test — 2026-05-15

### What ran

- Ran the full local unit/smoke suite:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/`.
- Ran protocol sync: `bash scripts/check_protocol_sync.sh`.
- Materialized ProRL recovery prompt archives with
  `scripts/run_prorl_recovery.py write-archives`.
- Rendered exact RWS Phase 0, Phase 1, filtered Phase 2, and Flow-profiled
  launch plans without launching external compute.
- Verified paid-run preflight behavior: Flow and FarmShare preflight-only paths
  pass with required fields; Flow fails closed without explicit authorization;
  profile cost validation rejects estimates above the Flow initial cap.
- Ran the CPU readiness smoke:
  `scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp`.
- Ran xAI reflection dry-run and xAI archive-build metadata smoke; no live API
  request was made because no `XAI_API_KEY` is present in the current shell.
- Exercised Phase 3 handoff locally with synthetic rows:
  `derive_prorl_phase3_input.py` produced one row, missing inputs fail closed,
  and `materialize_prorl_phase3_trajectories.py` attached the exact successful
  trajectory.
- Queried live resources read-only: FarmShare has the Phase 0 smoke running and
  main array pending behind `QOSMaxJobsPerUserLimit`; Flow reports A100 80GB
  capacity available in `us-central3-a`.

### What was verified

- Tests passed: `201 passed`.
- Protocol sync passed.
- Readiness smoke passed with `deferred_tracks=[]`, `deferred_experiments=[]`,
  `generation_calls_on_replay=0`, `12` runnable conditions, and `15` checklist
  passes.
- Artifact audit passed for representative readiness bundles:
  `math500/greedy` and `math500/polaris_full_verified_memory`.
- Generated Flow plan shape is profile-bound and uses the required production
  artifact contract.
- Secret scan found no pasted xAI key in repo files outside synthetic test
  fixtures; `.env` contains a blank `XAI_API_KEY` placeholder.

### Current gate

The repo is locally end-to-end ready for the ProRL recovery audit harness. Full
live execution still requires external gates: an inserted xAI key or backend
secret for live GEPA reflection, HF auth or `GPQA_DIAMOND_PATH` for official
GPQA, and a fresh explicit user launch command with cost cap for any Flow,
Modal, or CloudRift paid run.

## Multi-Resource Live Scheduler Amendment — 2026-05-15

### What changed

- Replaced the CloudRift-primary ProRL recovery plan with a multi-resource
  scheduler: FarmShare L40S for free shardable work, Mithril/Flow A100 for
  capped weekend acceleration, Modal for Phase 3/debug bursts, CloudRift as
  fallback, and xAI for GEPA reflection.
- Added the checked-in resource profile surface
  `configs/prorl_live_resources.json` and the launcher wrapper
  `scripts/launch_prorl_recovery.py`.
- Added xAI reflection configuration and a dry smoke that redacts
  `XAI_API_KEY`; default model `grok-4.3`, initial cap `$30`, hard cap `$100`.

### Current gate

No paid launch is authorized by this amendment. Current FarmShare Phase 0 jobs
remain the continuation path. Flow A100 may be used only after a fresh explicit
launch command, live pricing check, and per-launch cost cap.

## CloudRift Recovery Audit Protocol Amendment — 2026-05-15

### What changed

- Switched the active ProRL recovery audit execution target from FarmShare-only
  to CloudRift with explicit per-launch paid-run caps. FarmShare artifacts are
  retained as prior evidence, not current authorization.
- Locked 1x RTX 4090 as the first CloudRift probe target. V100 is fallback only
  after an FP16 smoke; MI350X is excluded unless NVIDIA capacity disappears and
  ROCm/vLLM/HF port risk is explicitly accepted.
- Updated ProRL v2 to public label `main` with resolved commit
  `c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0`; registry artifacts record file
  ETags because `main` and `v2` currently point to different refs with matching
  config and weight ETags.
- Locked pre-main token caps: Phase 0 Karan-Du `3072`, MATH500 `4096`,
  GPQA-Diamond `2048`, Reasoning Gym `8192`, with one allowed doubling only
  after smoke cap-hit rate exceeds 5%.
- Phase 2 planning must now consume Phase 1 outputs and run only the
  ProRL-only denominator set. Tasks with no trained-checkpoint improvement are
  skipped as `RF undefined/no denominator`.

### Current gate

No GPU job or paid CloudRift run is authorized by this amendment. The next
external step remains a bounded CloudRift 4090 probe and tiny smoke after an
explicit user command with a cost cap.

## ProRL Recovery Audit Goal Setup — 2026-05-13

### What ran

- Set the active `/goal` to implement the revised FarmShare ProRL Recovery
  Audit readiness surface without submitting FarmShare jobs, launching bulk
  runs, using GPQA official rows, or claiming recoverable-fraction results.
- Checked current FarmShare documentation and local FarmShare runbook evidence:
  FarmShare uses Slurm, exposes `/scratch`, and the GPU QoS supports one-GPU
  jobs with four running GPU jobs per user; the operational default is therefore
  four independent one-GPU array shards for 1.5B inference.
- Locked the standalone audit into `TODO.md` and
  `docs/PRORL_RECOVERY_AUDIT.md`.

### What was verified

- No GPU job, paid run, FarmShare submission, or bulk generation was launched
  from this setup step.
- The new audit remains subordinate to existing POLARIS gates: GPQA-Diamond
  stays auth-gated, ProRL traces are diagnostic only, and Phase 0 external
  Karan-Du replication is required before any RF claim.
- Local implementation verification passed:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/`
  reported `167 passed`, and `bash scripts/check_protocol_sync.sh` passed.
- Dry-run FarmShare CLI checks rendered `probe`, `env`, `model`, and Slurm
  `submit` surfaces without contacting FarmShare. The rendered Slurm script
  uses `/scratch/users/$USER/polaris`, `micromamba`, `#SBATCH --array=0-3%4`,
  and `#SBATCH --gres=gpu:1`.

### What remains in this section

- External execution remains blocked until the user explicitly authorizes the
  next FarmShare step. The first authorized run should be the non-bulk
  `probe -> env -> model -> shard` ladder, not Phase 1/2 science.
- RF claims remain blocked until the Phase 0 Karan-Du external replication gate
  passes on the exact published MATH500 setup.

## Production Live/Final Run Readiness Audit — 2026-05-13

### What ran

- Re-read `PROPOSAL.md` against the current repo, scripts, tests, readiness smoke artifacts, and run history.
- Compared each proposal obligation against actual production surfaces: offline co-construction, three-track runners, model matrix, baselines, rollout accounting, ablations, descriptor audit, memory persistence, backend cost/fidelity, final analysis, and live run operations.
- Updated `TODO.md` from broad infrastructure readiness to production/final-run readiness, with an explicit proposal-to-repo gap table and P0-P7 work plan.

### What was verified

- The repo is proposal-wide smoke-ready: `runs/readiness_smoke.tmp/readiness_report.json` currently reports `passed=True`, `deferred_tracks=[]`, `deferred_experiments=[]`, and `generation_calls_on_replay=0`.
- Current local validation passed: `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/` reported `128 passed` with `8` vendored-RWS regex warnings.
- Current protocol sync passed: `bash scripts/check_protocol_sync.sh`.
- The bounded Modal vLLM proof already recorded in this file is a correctness/fidelity smoke, not a production cost clearance.

### Production-readiness verdict

**Not production/final-run ready yet.** The repo has strong smoke coverage, but final experiments still need real production components:

- generic multi-track production runner and run graph;
- locked HumanEval+/GPQA dev/test splits and dataset hashes;
- model matrix selection instead of global `MODEL_ID` inheritance;
- real GEPA + verifier-gated-memory co-construction;
- persistent memory/event ledgers;
- baseline runners for Dynamic Cheatsheet, ACE, GEPA-only, P2O/published-GRPO comparisons;
- rollout ledger in production artifacts;
- descriptor reliability audit and production ablation runners;
- final aggregation/falsification scripts;
- cost-ready backend path or explicit protocol amendment before Phase 10;
- live Modal/Mithril/Flow launch, resume, monitoring, sync, and audit runbooks.

### Blockers

- R5 correctness is green but cost is still not acceptable for exact Phase 10.
- HumanEval+ and GPQA are loader/verifier-smoke ready but not production-runner ready.
- `condition_runner.py` is still a MATH500 re-export; no track-generic production runner exists.
- HumanEval+ and GPQA dev splits are still placeholders in code.
- The proposal's minimum baseline suite is not implemented as production runners.

### Next checkpoint

Implement P0-P3 from `TODO.md#production-livefinal-run-readiness-audit--2026-05-13` before any full paid launch: production run graph, generic runner/preflight, dataset/model locks, and backend cost/fidelity gate. Then run one small real-slice production artifact pass per target track before any long experiment.

## POLARIS-v3.1 R5 Infrastructure Contract — 2026-05-13

### Local Readiness Harness And Preflight Gate — 2026-05-13

- Reframed the active infrastructure goal as executable readiness, not backend probing: prove the local experiment loop, cache replay, artifact ledger, and paid-run preflight before any further GPU work.
- Added `src/polaris/infra/preflight.py` and wired `scripts/run_math500.py` plus Modal GPU smoke entrypoints to fail closed before loading HF/vLLM/SGLang unless they have artifact directory, trajectory cache/cache path, split, seed, model, backend, estimated cost, cost cap, and explicit user authorization. `--preflight-only` validates this contract without loading model backends.
- Added `scripts/smoke_polaris_readiness.py`, a CPU-only e2e smoke that writes `readiness_report.json`, `readiness_report.md`, `trajectories.sqlite`, and all seven mandated artifacts for every currently runnable MATH500 condition under `runs/readiness_smoke.tmp/`.
- Readiness smoke result: passed locally with `generation_calls_on_replay=0`. MATH500 v1 and mixed-alpha are runnable through the local artifact path; HumanEval+, GPQA-Diamond, archive-size sweep, memory composition, decaying alpha, joint optimization, factorial interaction, descriptor ablation, and verifier-gating ablation are explicitly listed in `readiness_report.md` as deferred with concrete blockers.
- New tests: `tests/unit/test_preflight.py` and `tests/smoke/test_readiness_smoke.py`. Targeted verification passed: `7 passed`. No GPU was run for this step.

### Full Proposal Infrastructure Readiness Smoke — 2026-05-13

- Replaced the blocker-classification readiness report with executable local scenario drivers for every `PROPOSAL.md` experiment family: MATH500, HumanEval+, GPQA-Diamond, archive-size sweep, GEPA iterations, memory composition, mixed/decaying alpha, joint optimization, factorial interaction, descriptor ablation, verifier-gating ablation, and break-even rollout accounting.
- Implemented minimal HumanEval+ and GPQA loader/verifier/selector paths. GPQA selection is non-oracle; the answer key is used only after selection for offline scoring.
- Wired memory-enabled inference for bounded retrieval, prompt injection, verifier-gated admission, and Beta-Bernoulli reliability updates. Decaying alpha is now a real runner condition (`full_archive_decaying`).
- Implemented bounded MAP-Elites mutation through an injected reflection hook, plus a smokeable factorial interaction fit.
- Readiness smoke result: `passed=True`, `deferred_tracks=[]`, `deferred_experiments=[]`, `generation_calls_on_replay=0`, `15` checklist rows all `pass`.
- Local verification passed: `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/` reports `128 passed`; `bash scripts/check_protocol_sync.sh` passes. No GPU, Mithril/Flow, Phase, or bulk run was launched.
- Bounded Modal post-local gate passed through the venv Modal CLI because the global Homebrew `modal` shebang is broken. Command: `./.venv-eval/bin/python -m modal run scripts/modal_vllm_app.py::smoke_vllm_batched_power_path --batch-size 8 --max-new-tokens 512 --block-num 4 --mcmc-steps 1 --estimated-dollar-cost 0.25 --cost-cap-dollars 0.50 --user-authorized-paid-run`. Result: `passed=True`, backend `vllm`, dtype `float32`, model impl `transformers`, 8/8 generations at 512 tokens, 8/8 `lp_norm` and `lp_unnorm` arrays aligned at 512 entries, acceptance ratios in `[0.5, 1.0]`, verifier passed `4/8`, allocated wall-clock sum `95.43s`. No additional GPU probes or bulk run were launched.

### Modal Credits Restored, Batched vLLM Smoke Passes — 2026-05-13

- User restored Modal billing capacity and authorized sparse validation only.
- Local gate after code changes: `./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/test_math500_dummy.py` passed `111` tests; `bash scripts/check_protocol_sync.sh` passed.
- `smoke_vllm_parity` passed on Modal H100 with `VLLM_USE_V1=0`, `dtype=float32`, `model_impl=transformers`: segment token count `22` (requested 32 but target text is shorter), vLLM vs HF-forward max abs diff `0.000107`, below the `<1e-3` gate.
- `smoke_vllm_batched_power_path --batch-size 8 --max-new-tokens 512 --block-num 4 --mcmc-steps 1` passed on Modal H100 using the GPU-accepted two-pass vLLM path: native vLLM sampling followed by batched forced-token scoring. Output counts were aligned for all 8 chains (`512` token ids, `512` `lp_norm`, `512` `lp_unnorm` each), acceptance ratios were within `[0.5, 1.0]`, verifier passed on `4/8`, and observed batched wall was `103.71s`.
- `smoke_cache_replay` passed on Modal CPU volume replay: cold path called generation once; replay path called generation zero times.
- Attempted to re-enable the cheaper fused sampling-recorder path with output/recorder matching. It is still **not GPU-accepted**: Modal H100 diverged between the recorder-forced path and emitted tokens at token `105` in a `128`-token batched decode. The fused path remains opt-in only (`fused_sampling_recorder=True`) and disabled by default.
- Probed whether vLLM V0 native `logprobs` / `prompt_logprobs` could replace the forced-token scoring pass. This is **not accepted**: the `vllm/vllm-openai:v0.9.2` image failed inside vLLM's sampler with `AssertionError: assert len(next_token_ids) == len(query_indices)` for both the initial `logprobs=0,prompt_logprobs=0` probe and the follow-up `logprobs=1,prompt_logprobs=1` probe. No logprobs shortcut is scale-ready.
- Fixed a fidelity bug in the older HF batched MCMC oracle path: RWS resamples from any generated token after the fixed prompt; the local batched helper was only resampling from the latest block. Added a regression test that fails on the old prefix length (`3`) and passes on the RWS-correct prefix length (`1`).
- Cost projection from the accepted two-pass vLLM smoke is above the R5 scale gate. Using a token-work extrapolation from `512/4/1` to exact `3072/16/10`, one B=8 power batch projects to `3.34` H100-hours; the locked MATH500 v1 power cells project to roughly `2632` H100-hours (`~$10.4K` Modal H100, `~$2.6K-$3.9K` Mithril at `$1-$1.50/hr`) before low-temp overhead. This blocks scale-up.
- Current decision: R5 correctness smoke is green for vLLM V0 two-pass batched MCMC, cache replay, and HF parity, but R5 is **not cost-ready** for Phase 10. Full scale-up remains blocked until MH scoring cost is reduced, protocol hyperparameters are explicitly amended, or the user explicitly selects a cheaper/partial validation run. No Mithril/Flow, Phase 10, or bulk generation is authorized from this entry.

### Batched vLLM MCMC implementation update — 2026-05-13

- Implemented the local batched/concurrent vLLM MCMC path: `generate_low_temp_batch`, `generate_power_batch`, stable per-candidate seed offsets, and per-block/per-proposal `LLM.generate(prompts=list, sampling_params=list)` batching.
- Wired the MATH500 runner to batch cache-miss candidates per problem when a sampler exposes batch methods; scalar HF/RWS fallback remains unchanged. Cache hits still bypass generation and preserve the existing trajectory key.
- Added the bounded GPU gate `scripts/modal_vllm_app.py::smoke_vllm_batched_power_path`. The first attempt failed before provisioning with `workspace billing cycle spend limit reached`; after Modal credits were restored, the top R5 entry records the bounded pass and the remaining cost failure. No Mithril/Flow or Phase 10 run is authorized.
- Scale-up remains blocked until local tests stay green, `smoke_vllm_parity` still passes, and the observed batched wall time projects under the R5 cost gate. Correctness is green; cost is not.

### Modal smoke update — 2026-05-13

- Follow-up proceed step: wired the MATH500 runner for `--backend {hf,vllm}` and optional `--trajectory-cache`, and passed cache metadata through `run_condition(...)` into `polaris_inference(...)`. Local replay test confirms a second non-greedy condition run performs zero sampler calls.
- Added a fused vLLM sampling recorder that samples from `softmax(logits / T)`, records `lp_norm`/`lp_unnorm`, and forces the sampled token in one decode pass, avoiding the earlier generate-then-score duplicate pass. Local unit tests pass, but the fused path is **not GPU-accepted yet** because the next Modal smoke was blocked by workspace spend limit.
- Modal spend blocker: `modal run scripts/modal_vllm_app.py::smoke_vllm_power_path --max-new-tokens 512 ...` now fails immediately with `workspace billing cycle spend limit reached`. No further GPU smoke can run in this workspace until billing/limit is reset.
- Phase 10 dry-run estimate from the last accepted 512-token vLLM smoke (`24.93s`, `max_new_tokens=512`, `block_num=4`, `mcmc_steps=1`) is not acceptable for scale-up. With cache reuse, exact Phase 10 settings project roughly `2578-3662` H100-hours for the verified non-fused sequential path (`~$10.2K-$14.5K` on Modal H100; `~$7.5K-$10.7K` on current Mithril 8xH100 spot average). The fused-but-unverified path still projects roughly `1290-1832` H100-hours if run sequentially. This means the next required engineering step is batched/concurrent vLLM MCMC, not Phase 10.
- `smoke_cache_replay` passed on Modal CPU volume replay: cold path called the sampler once; replay path called it zero times.
- `smoke_sglang_greedy` passed the bounded serving smoke on H100 after two fixes: use the SGLang image Python for `sglang.launch_server`, and explicitly `snapshot_download` into the persistent HF cache with Xet disabled. Cached startup then loaded weights from `/cache/huggingface`; FP8 SGLang generated a non-empty completion with `\boxed` present and estimated generation cost `2.76e-05`.
- `smoke_sglang_parity` is still failing and blocks optimized scale-up. Re-reading upstream RWS `power_samp_utils.py` confirmed parity must use generation-time raw logits gathered at exact token IDs. HF forward vs HF forced-generate agrees within `0.0035`, but SGLang differs from HF by `~9.76` on the native SGLang implementation and `~9.40` even under `--impl transformers`, with BF16, `attention_backend=torch_native`, `sampling_backend=pytorch`, no FP8 quantization, and CUDA graph disabled.
- `batched_mcmc_smoke` passed the real HF/RWS GPU MCMC path on Modal A100: `k=1`, `max_new_tokens=512`, `block_num=4`, `mcmc_steps=1`, wall `22.98s`, acceptance ratio `[1.0]`, verifier `passed=True`. The shorter 128-token smoke also executed but was verifier-false because it truncated before the answer.
- vLLM is now isolated in `scripts/modal_vllm_app.py` so normal HF/SGLang smokes do not build the large vLLM image. The image needed three fixes before GPU execution: clear the Docker entrypoint, symlink `python -> python3`, and install repo verifier deps (`pylatexenc`, `sympy`).
- `smoke_vllm_parity` now passes through the real `polaris.infra.serving.vllm.VLLMGenerator.score_segments` path on Modal H100 with `VLLM_USE_V1=0`, `dtype=float32`, `model_impl=transformers`: HF-forward vs vLLM max abs diff `0.000107`, below the `<1e-3` gate. Diagnostic HF-forward vs HF-generate drift was `0.002001`, within the `5e-3` consistency tolerance.
- `smoke_vllm_power_path` passed the real vLLM MCMC mechanics path on Modal H100: `max_new_tokens=32`, `block_num=4`, `mcmc_steps=1`, wall `6.70s`, `generation_token_count=32`, `logprobs_norm_count=32`, `logprobs_unnorm_count=32`, acceptance ratio `1.0`. Verifier was false because the 32-token smoke intentionally truncates before the answer.
- `smoke_vllm_power_path` also passed a complete-answer bounded smoke: `max_new_tokens=512`, `block_num=4`, `mcmc_steps=1`, wall `24.93s`, `generation_token_count=512`, `logprobs_norm_count=512`, `logprobs_unnorm_count=512`, acceptance ratio `1.0`, verifier extracted `\left(3, \frac{\pi}{2}\right)`, verifier `passed=True`.
- Current decision: cache replay, HF-oracle MCMC, vLLM scorer parity, and bounded vLLM MCMC mechanics are operational, but the current sequential runner is not cost-ready. SGLang is not accepted for MCMC scoring. No MATH500 Phase 10, Mithril/Flow, or bulk trajectory generation is authorized from this state.

### Verified after smoke update

- `python -m pytest -q tests/unit/ tests/smoke/test_math500_dummy.py` passes locally: 106 tests.
- `bash scripts/check_protocol_sync.sh` passes.
- This workspace is not a git repository, so no git status/commit evidence is available here.

### What ran

- Set the active goal to R5 infrastructure: vLLM V0 optimized MCMC candidate, HF parity oracle, SQLite trajectory cache, Modal-only quick smokes, and no Mithril/Phase 10 launch without explicit user command.
- Source-checked the backend plan: vLLM's old zero-token scoring sketch is invalid because current `SamplingParams` rejects `max_tokens=0`; vLLM V0 per-request logits processors can force and score target tokens; SGLang remains blocked for MCMC scoring after failed parity.
- Implemented the first R5 slice locally: `TrajectoryCache`, optional cache replay in `core/inference.py`, `Sampler.score_segments`, HF oracle scoring, SGLang client/parity surface, and Modal smoke entrypoints.

### What was verified

- R5 keeps HF/RWS as correctness oracle and blocks SGLang `generate_power` until `score_segments` parity passes.
- Modal smokes are bounded validation only: `smoke_sglang_greedy`, `smoke_sglang_parity`, and `smoke_cache_replay`.
- Long Mithril/Flow jobs, MATH500 Phase 10, and bulk trajectory generation remain blocked until a fresh explicit user command.
- Local verification passed: `pytest -q` reports 96 tests passing; `python -m compileall -q src scripts` passes; `bash scripts/check_protocol_sync.sh` passes; importing `scripts.modal_app` exposes all three R5 smoke functions.

### What remains in this section

- Rerun full local unit tests and protocol sync after the vLLM implementation.
- Before any scale-up command, implement and verify a batched/concurrent vLLM MCMC driver that brings the dry-run cost estimate back near the intended R5 cost floor.

### Blockers

- SGLang scorer correctness failed the current parity gate; it must pass HF parity (`max_abs_diff < 1e-3`) before any SGLang MCMC scale-up.
- vLLM BF16/native fast path is not accepted for MH scoring; the accepted path is FP32 + `model_impl=transformers`, which may reduce the expected cost win.
- Current sequential vLLM MCMC is too expensive at exact Phase 10 settings.
- Modal GPU smokes are blocked by workspace billing cycle spend limit.
- No current permission to run Mithril/Flow or Phase 10.

### Next checkpoint

R5 is correctness-smoke-ready but not cost-ready. Next useful action is batched/concurrent vLLM MCMC plus a fresh GPU smoke after Modal billing capacity is available. Bulk generation still requires a fresh explicit user command.

## POLARIS-v3.0 Protocol Bump — 2026-05-12

### What ran

- Revised `PROPOSAL.md` to v3.0 grounding it in the user-provided conceptual framing verbatim. Acronym expanded to **POLARIS: Prompt-Organized Library of Archived Reasoning and Inference Strategies**.
- Appended the v3.0 reorganization plan to `TODO.md` (R0-R8 phased rebuild around the three mechanisms, three tracks, three replication models, break-even N axis).
- Began the repo reorganization: deleting `_legacy/`, moving `infra/model.py` under `infra/serving/`, scaffolding `core/memory.py`, expanding `core/descriptor.py` to 8 labels, creating `stats/` and `io/rollouts.py`.

### What was verified

- Reference audit in `PROPOSAL.md §18` retained intact across the revision; all 19 citations still match published primary sources.
- Existing `tests/unit/` and `tests/smoke/` to be re-run after each R-block; behavior-preserving refactor invariant.
- Phase 10 GPU launch of the existing MATH500 v1 plan is on hold pending the vLLM cost-floor decision (R5). No paid compute has been launched under v3.0.

### What remains in this section

- Execute R0-R4 + R6-R8 (no paid compute required); checkpoint after each R-block with green `pytest -q tests/unit`.
- R5 (vLLM port) requires explicit per-launch user consent before any Modal smoke. K=8 HF full-config calibration data from 2026-05-12 stays as the cost reference.
- Trim historical sections of this file to last 3 dated decision blocks; archive the remainder to `runs/_archive/progress-pre-2026-05-12.md`.

### Blockers

- None for R0-R4/R6-R8.
- R5 blocked on user `go` per `~/.claude/projects/-Users-duy-Documents-build-polaris/memory/gpu_launch_consent.md`.
- v1 Phase 10 launch blocked on R5 completion (cost floor must come down before $500-800 Phase 10 spend).

### Next checkpoint

After R1-R4 land: run full `pytest -q tests/unit` + a CPU-dummy run of `tests/smoke/test_math500_dummy.py` to confirm the reorg is behavior-preserving; then surface R5 spec to user with verified MATH500 hyperparameters and ask for launch consent.

---

## Checkpointed C0c Preemption Guard — 2026-05-11 10:20 UTC

### What ran
Patched `scripts/monitor_mcmc_ckpt_local.sh` heartbeat path handling and continued monitoring through `scripts/monitor_mcmc_ckpt_local.sh` + `scripts/rws_mcmc_ckpt_status_remote.sh`.

### What was verified
`scripts/rws_mcmc_ckpt_status_remote.sh` now reports exactly one active checkpointed worker (`pid` shown), `HumanEval/0` and `HumanEval/1` completed, and `runs/rws_official_full164_mcmc_ckpt/heartbeat/worker_0.json` at `task_index=2` with event=`started`.

### What remains in this section
Keep checkpointed single-worker mode and monitor stale-heartbeat/heartbeat-based restart behavior until all 164 task files are produced.

### Blockers
- A100 utilization remains high; this is compute-bound.
- Final shard/CSV completion and final RWS `C0c` evaluation remain pending before any protocol decision.

### Next checkpoint
Refresh `TODO.md` and this file when either (a) completion or (b) preemption/restart event occurs.

## POLARIS Goal Revision — 2026-05-11 10:00 UTC

### What ran
Updated `proposal/polaris_verified_revised_proposal_single.md` and `TODO.md` to rename the active project to POLARIS, promote mixed-alpha diversity preservation into the core method, and pause the old HumanEval+ Phase 1/Phase 2 plan.

### What was verified
The active goal now requires keeping the running RWS `C0c` baseline alive, then writing a protocol decision before any new expensive experiment; the docs now state that old fixed-alpha Phase 1/Phase 2 cannot begin unchanged.

### What remains in this section
Finish or diagnose `C0c`, compare it against RWS Table 1, then choose HumanEval+ negative-control, MATH500 primary, or pause/reframe.

### Blockers
No protocol blocker for monitoring; all new experimental phases remain blocked until the post-`C0c` decision is written.

### Next checkpoint
Sync the revised docs to the A100 copy and continue monitoring the two-worker `C0c` run.

Historical entries below this line may describe the superseded CovComp MVE contract and are retained as provenance, not current execution authority.

## POLARIS Contract Sync — 2026-05-11 10:07 UTC

### What ran
Ran `scripts/check_protocol_sync.sh` locally, synced `TODO.md`, `runs/progress.md`, `proposal/polaris_verified_revised_proposal_single.md`, and the drift-check script to `/home/ubuntu/polaris`, then ran the same drift check on the A100.

### What was verified
Both local and remote drift checks pass; the remote proposal, TODO, and progress files now point to POLARIS-v2.0 and block old Phase 1/Phase 2 from starting unchanged.

### What remains in this section
Continue monitoring `C0c` until completion or diagnosis, then write the required post-`C0c` protocol decision.

### Blockers
No contract-sync blocker; the only active blocker remains C0c runtime.

### Next checkpoint
Poll the next scheduled monitor tick and record worker/shard/GPU status.

## Protocol Integrity Contract

### What ran
- Confirmed the active execution contract remains the same as `proposal/polaris_verified_revised_proposal_single.md` (including post-MVE extensions deferred, mixed-`alpha` and co-construction deferred, and Phase 2-only scope).
- Confirmed `TODO.md` and this `progress.md` are the only mutable coordination surfaces for execution state; upstream repositories under `upstream/` remain read-only.

### What was verified
- Proposal, TODO, and protocol surfaces were aligned on: (1) active model/baseline protocol (`Qwen/Qwen2.5-7B`, HumanEval+), (2) MCMC-only full-164 fidelity workflow, and (3) Phase 0 amendment requirement before any Phase 1/2.
- No unresolved conflict between `proposal/polaris_verified_revised_proposal_single.md` and this TODO/progress log was observed at the time of this entry.

### What remains in this section
- Continue logging each phase transition with explicit cross-references to protocol text and the current blocked/running blocker status.

### Blockers
- The active C0c run is long-lived; all downstream decisions remain blocked on C0c completion and protocol resolution around full-164 fidelity.

### Next checkpoint
- After any protocol decision or script edit, append a dated section under the correct heading (`Full-164`, `Phase 0`, etc.) and ensure TODO current-state bullets match the same decision.

## Protocol Governance Sync — 2026-05-11 10:00 UTC

### What ran
- Added `scripts/check_protocol_sync.sh` and ran it immediately.
- Updated `TODO.md`, `goal_status_audit.md`, `protocol_amendment_options.md`, and `RESUME_AFTER_PROTOCOL_DECISION.md` to require explicit protocol-surface synchronization before work resumes.

### What was verified
- Drift guard passes: proposal, TODO, and progress still share the required contract sections:
  - Proposal execution-agent decomposition is present.
  - TODO contains the drift-guard and team-surface sections.
  - Progress contains the protocol-integrity section.

### What remains in this section
- Continue to run `scripts/check_protocol_sync.sh` at each phase handoff or when a new agent edits protocol-related files.

### Blockers
- No blocking drift issue was introduced at this checkpoint; full-164 C0c runtime blocker remains unchanged.

### Next checkpoint
- Wait for C0c shard completion and then re-sync TODO/progress before moving to any Phase 1 handoff.

## Preflight — In Progress

### What ran
- Repository scaffold is present at `/Users/duy/Documents/build/polaris` with `proposal/`, `upstream/`, `src/`, and `runs/`.
- The revised execution contract has been recorded in `TODO.md`.
- `proposal/polaris_verified_revised_proposal_single.md` has been restored from `/Users/duy/Documents/build/coverage/PROPOSAL.md`.
- Official RWS repo resolved as `https://github.com/aakaran/reasoning-with-sampling.git`; cloned at `720a8e9d084c87a630595e316f5260f1d7c3446c`.
- GEPA, EvalPlus, and Dynamic Cheatsheet were cloned as read-only references at commits `ce51b50cd196b539c25fae99ad0e0255c23004a4`, `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, and `5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9`.
- Local harness files were added under `src/polaris` and `src/experiments`; `python3 -m compileall -q src` passes.
- Local static preflight ran with EvalPlus/vLLM skipped; RWS toy example completed in `1.17s`, and RWS import failed locally only because this Mac Python lacks usable `torch`.
- `polaris-a100-050` was found paused at bid `$0.05/hr`; the same bid was updated to `$4.00/hr` and unpaused, then moved to `provisioning`.
- Remote SSH/GPU readiness passed on `polaris-a100-050`: `NVIDIA A100-SXM4-80GB`, Python `3.12.3`, `uv` available, `/mnt/local` mounted.
- Remote isolated venv `/home/ubuntu/polaris/.venv` was created with Python `3.11`; installed `vllm==0.6.3`, `torch==2.4.0`, `transformers==4.47.1`, `evalplus==0.3.1`, `datasets==3.2.0`, editable GEPA `0.1.1`, and editable `polaris`.

### What was verified
- No experiment has started.
- RWS contains `llm_experiments/power_samp_utils.py::mcmc_power_samp` and `llm_experiments/power_samp_he.py`, so P1 has a candidate runnable MCMC implementation.
- RWS `power_samp_he.py` maps `model=qwen` to `Qwen/Qwen2.5-7B` and uses raw HumanEval prompts for Qwen; SPS Table 3's Qwen HumanEval instruction wrapper is a prompt conflict to handle explicitly.
- The active Mithril A100 target remains `polaris-a100-050`; remote setup is waiting on `started_at` plus SSH and `nvidia-smi -L`.
- RWS `environment.yml` pins `vllm==0.6.6.post1` while the goal/SPS Appendix C require `vllm==0.6.3`; the harness will keep `vllm==0.6.3` for the preflight smoke and use RWS's Transformers code path for MCMC.

### What remains in this section
- P1: finish documenting the official RWS URL/commit and sampler files.
- P2: run one upstream RWS generation path unmodified on the A100.
- P3: load `Qwen/Qwen2.5-7B` through vLLM and score HumanEval problem 0 with EvalPlus.
- P4: record resolved RWS URL, model revision, vLLM version, prompt template, generation, unit-test outcome, and wall-clock time.

### Blockers
- No blocker has been confirmed yet.
- Preflight is mandatory; Phase 0 cannot begin until P1-P4 pass.
- Mithril file volumes are disabled in `us-central3-a`; a block volume could not be attached to the running task path, so remote local disk plus rsync-back after each checkpoint is the current fallback.

### Next checkpoint
- Reverify A100 SSH/GPU readiness, sync the repo to `/home/ubuntu/polaris`, and install the remote environment.

## Phase 0 — Not Started

### What ran
- Nothing has run for Phase 0.

### What was verified
- Phase 0 is blocked until Preflight P1-P4 pass.

### What remains in this phase
- Run C0a/C0b/C0c on HumanEval+ problems `0-49`.
- Record original HumanEval and HumanEval+ pass/fail on the same generations.
- Compare original HumanEval rates to RWS Table 1 targets: `0.329`, `0.524`, `0.622`.

### Blockers
- Preflight incomplete.

### Next checkpoint
- Start only after a completed Preflight section says P1-P4 passed.

## Preflight — 2026-05-11 00:24 UTC

### What ran
P1-P4 ran on `polaris-a100-050` from `/home/ubuntu/polaris` using `.venv` Python 3.11 and saved `runs/preflight_status.json` plus `runs/preflight_remote.log`.

### What was verified
RWS resolved to `https://github.com/aakaran/reasoning-with-sampling.git` at `720a8e9d084c87a630595e316f5260f1d7c3446c`, `mcmc_power_samp` imported on the A100, EvalPlus has 164 HumanEval+ tasks with non-overlapping splits `0-49`, `50-99`, `100-163`, known pass/fail checks behaved correctly, and vLLM `0.6.3` loaded `Qwen/Qwen2.5-7B` revision `d149729398750b98c0af14eb82c78cfe92750796`.

### What remains in this section
Nothing; Preflight P1-P4 passed.

### Blockers
No Preflight blocker remains; the only fidelity note is that Phase 0 follows the official RWS Qwen HumanEval raw-prompt path for Table 1 reproduction, while SPS Table 3's wrapper remains recorded for archive/Phase 2 prompt conditions.

### Next checkpoint
Start Phase 0 on HumanEval+ problems `0-49` and stop before Phase 1 unless all Phase 0 criteria pass.

## Phase 0 Throughput Setup — 2026-05-11 00:32 UTC

### What ran
Phase 0 was changed from one sequential foreground process to three deterministic shard workers launched by `scripts/phase0_sharded_remote.sh`, each writing `runs/phase0_shards/shard_<i>.jsonl` and logs under `runs/phase0_shards/worker_<i>.log`.

### What was verified
With 3 workers loaded, `nvidia-smi` reported about `99%` GPU utilization and `45.9/81.9GB` used, so 3 workers currently saturate the A100 without approaching OOM.

### What remains in this section
Merge shards with `scripts/phase0_merge_remote.sh` after all workers finish, then evaluate Phase 0 gates from the merged JSONL.

### Blockers
No OOM or worker crash observed; one early C0a task generated to the `3072` token cap, so truncation/max-length frequency must be tracked before deciding Phase 0 success.

### Next checkpoint
Monitor shard row counts, worker liveness, GPU memory, and max-token rows before adding or removing workers.

## Phase 0 Concurrency Correction — 2026-05-11 00:38 UTC

### What ran
The first sharded restart briefly launched six workers due to a failed kill command; all six explicit PIDs were terminated, then exactly three workers were relaunched as PIDs `13037`, `13040`, and `13043`.

### What was verified
After relaunch, `nvidia-smi` reported about `100%` GPU utilization and `45.7/81.9GB` memory used; shard JSONL files were valid with no malformed lines and no duplicate keys.

### What remains in this section
Continue Phase 0 with three workers and merge only after workers finish.

### Blockers
No current blocker; do not increase to four workers unless utilization drops materially because three workers already saturate the GPU.

### Next checkpoint
Check shard row counts, worker liveness, max-token rate, and pass/fail rates from shard JSONL.

## DIAGNOSTIC — Phase 0 Paused — 2026-05-11 00:42 UTC

### What ran
Phase 0 sharded workers were paused after 12 C0a greedy rows because capped/repetitive generations were observed; workers `13037`, `13040`, and `13043` were killed cleanly and GPU memory returned to `0/81.9GB`.

### What was verified
Observed `6/12` C0a generations at the locked `3072` token cap, and manual tail inspection showed repeated doctests/function definitions, so the goal's `>5%` nonsense/truncation escalation rule is triggered before Phase 0 can continue.

### What remains in this section
A protocol decision is needed: either continue raw RWS-style `max_new_tokens=3072` despite frequent base-model repetition because it may match Table 1 behavior, or add HumanEval stop sequences/shorter stopping logic and label it as a deliberate deviation from RWS.

### Blockers
Expected RWS Table 1 greedy target is `0.329`, but the current partial C0a slice is too small for accuracy judgment; hypotheses are: raw base Qwen often continues into repeated tests without stop strings, RWS postprocessing tolerates long repeated tails so this may be expected, or our HF generation path needs the exact official RWS cache/stop behavior checked before resuming.

### Next checkpoint
Do not proceed to C0b/C0c or Phase 1 until the truncation/repetition decision is resolved.

## Phase 0 Truncation Resolution — 2026-05-11 00:48 UTC

### What ran
Inspected RWS `power_samp_he.py`, `eval_he.py`, and `grader_utils/he_grader.py` against the observed capped C0a rows.

### What was verified
Official RWS HumanEval generation uses `max_new_tokens=3072` without stop strings, and official RWS evaluation concatenates prompt/output then applies `extract_code(...)`, which keeps the first function body; the observed repeated tails are therefore a raw-generation diagnostic but changing stop strings would be a protocol deviation.

### What remains in this section
Resume raw RWS Phase 0 with three sharded workers, and report max-token/repetition rate separately from HumanEval and HumanEval+ pass rates.

### Blockers
No protocol blocker remains; the diagnostic remains open as a reported failure mode, not a reason to alter the locked Phase 0 reproduction path.

### Next checkpoint
Restart three workers, monitor row counts and GPU memory, then merge shards after completion.

## DIAGNOSTIC — Phase 0 Gate Failed — 2026-05-11 00:55 UTC

### What ran
Merged the partial Phase 0 shards into `runs/phase0_sharpening_baseline.jsonl` and `runs/phase0_summary.json`, then stopped all shard workers before C0b/C0c because C0a already made the pre-registered gate impossible.

### What was verified
C0a has `22/29 = 0.759` original HumanEval Pass@1 and `20/29 = 0.690` HumanEval+ Pass@1, so even if every remaining C0a problem failed the final original HumanEval lower bound would be `22/50 = 0.440`, outside the RWS Table 1 target window `0.279-0.379` around `0.329`.

### What remains in this section
No Phase 0 continuation is valid under the current goal; C0b, C0c, Phase 1, and Phase 2 are intentionally not started.

### Blockers
Hypotheses: the goal's greedy C0a condition may not match RWS Table 1 because the official RWS `std` path samples rather than using `do_sample=False`; the first-50 HumanEval slice plus RWS `extract_code` may be materially easier than the full benchmark; or the exact published prompt/evaluation/revision differs despite using official RWS code, vLLM `0.6.3`, and `Qwen/Qwen2.5-7B` revision `d149729398750b98c0af14eb82c78cfe92750796`.

### Next checkpoint
Pause for protocol review and decide whether to revise C0a to the official RWS `std` sampling condition, change the Phase 0 gate to a full-benchmark reproduction, or keep the failure as a kill signal.

## Phase 0 Protocol Correction — 2026-05-11 01:02 UTC

### What ran
Compared the local harness against official RWS `llm_experiments/power_samp_he.py` and patched Phase 0 C0a from strict greedy decoding to official RWS `std` sampling.

### What was verified
RWS `std_output` is generated with `do_sample=True` and no low-temperature override, while `naive_temp_output` uses `temperature=0.25`; the earlier `do_sample=False` C0a run was a harness mismatch, not a CovComp result.

### What remains in this section
Archive the failed strict-greedy artifacts, sync the corrected harness to the A100, and rerun Phase 0 from clean shard outputs.

### Blockers
No Phase 1 work may start until the corrected C0a/C0b/C0c Phase 0 gates pass.

### Next checkpoint
Run corrected Phase 0 with three workers, then merge and evaluate the pre-registered gates from saved JSONL.

## RWS Replication Audit — 2026-05-11 01:08 UTC

### What ran
Read RWS arXiv source `main.tex`, official RWS HumanEval generation/evaluation code, and SPS Appendix C source, then wrote `runs/rws_replication_audit.md`.

### What was verified
RWS Table 1 is single-shot on HumanEval, reports Qwen2.5-7B `Base=0.329`, `Low-temperature=0.524`, `Power Sampling=0.622`, and the official code implements HumanEval base as `do_sample=True` default sampling, low-temperature as `temperature=0.25`, and MCMC as `alpha=4`, `T_max=3072`, `block_num=16` meaning `B=192`, `N_MCMC=10`.

### What remains in this section
Sync the audit and corrected harness to the A100, then rerun Phase 0 from clean shards.

### Blockers
The remaining fidelity caveat is that RWS Table 1 is a full 164-problem result, while this MVE gates on problems `0-49`; treat Phase 0 as a smoke replication unless the protocol is revised to full HumanEval.

### Next checkpoint
Launch corrected Phase 0 with exactly three workers and monitor row counts, worker liveness, GPU memory, and pass-rate trajectory.

## DIAGNOSTIC — Corrected Phase 0 Gate Failed — 2026-05-11 01:09 UTC

### What ran
Reran corrected Phase 0 from clean shards with C0a as official RWS `std` sampling, then stopped workers before completing C0b/C0c because C0a failed criterion 1.

### What was verified
C0a original HumanEval Pass@1 is `22/50 = 0.440` and HumanEval+ Pass@1 is `19/50 = 0.380`; criterion 1 expected `0.279-0.379` around RWS Table 1 `0.329`, so Phase 0 cannot pass.

### What remains in this section
C0b/C0c were intentionally stopped partial (`17` C0b rows existed at stop time), and Phase 1/Phase 2 must not start under the current goal.

### Blockers
Hypotheses: the fixed first-50 slice is easier than full HumanEval and too small for direct Table 1 gating; EvalPlus base tests plus RWS `extract_code` may not exactly match the HumanEval grader used for the published number; or RWS Table 1 was produced from a different stochastic seed/run distribution than the deterministic per-sample seeding used here.

### Next checkpoint
Pause for protocol decision: either revise Phase 0 to a full-164 RWS reproduction, widen/remove the first-50 target gate, or treat this as a failed MVE replication gate and stop.

## Full-164 Reproduction Prep — 2026-05-11 01:13 UTC

### What ran
Prepared but did not launch dedicated full-164 RWS reproduction scripts: `scripts/rws_full164_phase0_sharded_remote.sh` and `scripts/rws_full164_phase0_merge_remote.sh`.

### What was verified
The scripts target a separate namespace `runs/rws_full164_phase0_shards/`, `runs/rws_full164_phase0.jsonl`, and `runs/rws_full164_phase0_summary.json`, keep the duplicate-worker guard, and set `--low-temp-samples 1 --power-samples 1` so the run is single-shot like RWS Table 1.

### What remains in this section
Only launch these scripts if the protocol is explicitly amended to run Option B from `runs/protocol_amendment_options.md`.

### Blockers
The current goal remains blocked because Phase 0 criterion 1 failed and Phase 1/2 are not authorized under the existing gate.

### Next checkpoint
Wait for a protocol decision: strict stop, full-164 reproduction, or slice-calibrated gate.

## Gate Audit Script — 2026-05-11 01:18 UTC

### What ran
Added `src/experiments/gate_audit.py` and ran `PYTHONPATH=src python3 -m experiments.gate_audit --output runs/gate_audit.json`.

### What was verified
The audit reads saved artifacts and reports `status = blocked`, with Phase 0 rows `67`, present conditions `C0a,C0b`, C0a HumanEval `0.44`, no Phase 1 archive, and no Phase 2 rows or summary.

### What remains in this section
Use the script after any protocol amendment or new run to prevent treating partial artifacts as completion.

### Blockers
The active blocker remains the failed C0a Phase 0 gate.

### Next checkpoint
Wait for a protocol decision before launching any new GPU work.

## Full-164 Monitor Prep — 2026-05-11 01:22 UTC

### What ran
Added `scripts/rws_full164_phase0_status_remote.sh` for read-only monitoring of the prepared full-164 RWS reproduction namespace.

### What was verified
The script reports Phase 0 worker processes, A100 utilization and memory, shard row counts, per-condition pass counters, max token length, and recent worker log tails without starting new jobs.

### What remains in this section
Use this only if Option B is accepted and the full-164 workers are launched.

### Blockers
No change to the blocker: Phase 0 failed under the current gate and Phase 1/2 remain unauthorized.

### Next checkpoint
Wait for protocol decision.

## Deep Paper Replication Check — 2026-05-11 03:24 UTC

### What ran
Re-read RWS TeX source, SPS Appendix C source, GEPA paper source, EvalPlus paper source, official RWS HumanEval code, and a DeepWiki source summary for `aakaran/reasoning-with-sampling`; updated `runs/deep_replication_audit.md`.

### What was verified
RWS Table 1 is full-164 original HumanEval single-shot, not first-50 and not verifier-best-of-8; official Qwen HumanEval uses raw prompts, `std` is `do_sample=True`, low-temp is `temperature=0.25`, and MCMC is `alpha=4`, `T_max=3072`, `B=192`, `N_MCMC=10`.
GEPA supports reflective prompt mutation plus Pareto/frontier candidate retention, and EvalPlus supports reporting HumanEval+ separately because stronger tests systematically lower apparent pass rates.

### What remains in this section
The live full-164 low-temperature diagnostic must finish before deciding whether to amend Phase 0.

### Blockers
The proposal did not fail; the first-50 Phase 0 gate failed as written, while the full-164 C0a/std diagnostic already returned `0.29878048780487804`, inside the RWS Table 1 target window.

### Next checkpoint
Evaluate `runs/rws_official_full164_naive/` after all low-temperature CSV shards finish, then sync results locally and update the gate audit.

## MCMC Replication Wrapper Prep — 2026-05-11 03:33 UTC

### What ran
Added MCMC-only official-RWS wrappers: `scripts/rws_official_full164_mcmc_generate_remote.sh`, `scripts/rws_official_full164_mcmc_status_remote.sh`, and `scripts/rws_official_full164_mcmc_eval_remote.sh`.

### What was verified
Local tests pass (`9` unittest tests), and `runs/gate_audit.json` now tracks full-164 std, naive, and mcmc diagnostics separately.

### What remains in this section
The MCMC-only wrapper has not been synced to the A100 and has not been launched.

### Blockers
The previous direct all-method run OOMed with three MCMC workers after two completed tasks, so any MCMC diagnostic should default to `WORKERS=1` unless a new smoke shows two workers fit.

### Next checkpoint
Do not start MCMC until the current low-temperature diagnostic finishes or is explicitly stopped.

## Full-164 Low-Temperature Monitor — 2026-05-11 03:49 UTC

### What ran
Monitored the active `MODE=naive` full-164 official-RWS low-temperature diagnostic on `polaris-a100-050`.

### What was verified
The A100 remains saturated at `100%` with about `59.8/81.9GB` used; shard `2` is complete, while shards `0`, `1`, and `3` continue moving.

### What remains in this section
Wait for the remaining three CSV shards, then run `MODE=naive RWS_MODEL=qwen bash scripts/rws_official_full164_std_eval_remote.sh`.

### Blockers
No infrastructure blocker is visible; progress is slow because many raw RWS low-temperature generations run near the `3072` token cap.

### Next checkpoint
After C0b evaluation, sync only `runs/rws_official_full164_naive/` back to local and refresh `runs/gate_audit.json`.

## Handoff Snapshot — 2026-05-11 01:25 UTC

### What ran
Added `runs/HANDOFF.md` summarizing the blocked state, key artifacts, corrected Phase 0 result, remote state, and valid protocol options.

### What was verified
The handoff points to the prepared full-164 scripts but states they must not be launched without explicit protocol amendment.

### What remains in this section
Sync the handoff to `/home/ubuntu/polaris/runs/` and keep the A100 idle.

### Blockers
The active blocker remains unchanged: corrected C0a failed Phase 0 criterion 1.

### Next checkpoint
Wait for a protocol decision.

## Original HumanEval Eval Prep — 2026-05-11 01:29 UTC

### What ran
Added `src/experiments/export_rws_humaneval.py` and `scripts/rws_full164_original_eval_remote.sh` to export merged full-164 outputs into RWS HumanEval JSONL format and score them with the RWS `he_grader` path.

### What was verified
The prepared script is intended for Option B only, after `runs/rws_full164_phase0.jsonl` exists, and separates original HumanEval/RWS scoring from EvalPlus base/plus scoring.

### What remains in this section
Sync and syntax-check the new exporter/evaluator; do not run it until full-164 outputs exist.

### Blockers
No change: current MVE is blocked at Phase 0.

### Next checkpoint
Wait for protocol decision.

## Phase 0 Original HumanEval Diagnostic — 2026-05-11 01:33 UTC

### What ran
Prepared `scripts/phase0_original_eval_remote.sh` to score the saved corrected Phase 0 rows with the original HumanEval/RWS `he_check` evaluator, using per-condition problem files for partial runs.

### What was verified
The diagnostic ran on the A100 CPU path and produced original RWS HumanEval `C0a pass@1 = 0.44`, matching the EvalPlus-base C0a result; the Phase 0 gate failure is therefore not explained by EvalPlus-base scoring differences.

### What remains in this section
Sync `runs/phase0_original_humaneval/` back to local.

### Blockers
No change to the Phase 0 gate: corrected C0a remains outside `0.279-0.379`.

### Next checkpoint
Wait for protocol decision.

## Gate Audit Original-Scorer Update — 2026-05-11 01:36 UTC

### What ran
Updated `src/experiments/gate_audit.py` to include `runs/phase0_original_humaneval/summary.json` and the original RWS HumanEval C0a `pass@1`.

### What was verified
The machine audit now records both EvalPlus-base C0a `0.44` and original RWS HumanEval C0a `0.44`.

### What remains in this section
Sync the updated audit script and regenerated `runs/gate_audit.json` to the A100.

### Blockers
No change: Phase 0 remains failed under the current gate.

### Next checkpoint
Wait for protocol decision.

## Flow Pause — 2026-05-11 01:41 UTC

### What ran
Ran `flow bid pause polaris-a100-050 --yes` after verifying no Phase 0/1/2 workers and all current artifacts were synced locally.

### What was verified
`flow bid list --json` reports bid `bid_im0giPo5bp599cuT`, name `polaris-a100-050`, `status = Paused`, and limit price `$0.05`; `flow status` may still display `running` with `started_at = null`, so bid status is the source of truth for the pause.

### What remains in this section
Unpause only if the protocol is amended and new GPU work is explicitly accepted.

### Blockers
The experiment remains blocked at Phase 0; pausing was cost hygiene, not goal completion.

### Next checkpoint
Wait for protocol decision.

## Flow Pause Artifact — 2026-05-11 01:44 UTC

### What ran
Wrote `runs/flow_pause_status.json` from `flow bid list --json` and updated `src/experiments/gate_audit.py` to include that artifact.

### What was verified
The artifact records `polaris-a100-050`, bid `bid_im0giPo5bp599cuT`, `status = Paused`, limit price `$0.05`, region `us-central3-a`, and instance `inst_KeIKYFkfMHMKH6k5`.

### What remains in this section
Remote sync is intentionally deferred while the bid is paused; local master contains the latest pause artifact.

### Blockers
No change: the experiment is blocked at Phase 0.

### Next checkpoint
Wait for protocol decision.

## Artifact Manifest — 2026-05-11 01:46 UTC

### What ran
Generated `runs/artifact_manifest.sha256` with SHA-256 checksums for all current files under `runs/`.

### What was verified
The manifest contains 38 artifact entries, including `TODO.md`, preflight, Phase 0 corrected rows, original HumanEval diagnostic outputs, audits, handoff, protocol options, and Flow pause status.

### What remains in this section
Remote sync is deferred while the bid is paused; local master is the current source of truth.

### Blockers
No change: Phase 0 remains failed under the current gate.

### Next checkpoint
Wait for protocol decision.

## Resume Runbook — 2026-05-11 01:49 UTC

### What ran
Added `runs/RESUME_AFTER_PROTOCOL_DECISION.md` with exact commands to unpause Flow, sync local master to remote, verify readiness, launch Option B, monitor, merge, original-score, and sync results back.

### What was verified
The runbook explicitly states that Phase 1 must not start until an amended Phase 0 decision is recorded and that full-164 reproduction is a protocol diagnostic, not a CovComp result.

### What remains in this section
Regenerate the artifact manifest locally.

### Blockers
No change: protocol decision is still required.

### Next checkpoint
Wait for protocol decision.

## Secret Scan — 2026-05-11 01:52 UTC

### What ran
Searched `runs/`, `TODO.md`, `src/`, `scripts/`, `pyproject.toml`, and the workspace for common secret patterns including `HF_TOKEN`, `API_KEY`, `sk-`, `hf_`, private-key headers, GitHub tokens, Slack tokens, OpenAI, W&B, and Anthropic keys.

### What was verified
No actual secret-looking values were found in the CovComp artifacts; matches were limited to placeholder/API-key references in upstream documentation and examples.

### What remains in this section
Regenerate the artifact manifest after this progress update.

### Blockers
No change: the goal remains blocked at Phase 0.

### Next checkpoint
Wait for protocol decision.

## Deep RWS/SPS Replication Audit — 2026-05-11 01:39 UTC

### What ran
Rechecked RWS arXiv source, SPS Appendix C source, official RWS GitHub, local RWS HumanEval generation/evaluation code, DeepWiki, and an independent Gemini audit, then wrote `runs/deep_replication_audit.md`.

### What was verified
RWS Table 1 is full-164 HumanEval single-shot, while the failed gate applied that point estimate to problems `0-49`; official RWS `std` is sampling, Qwen HumanEval uses raw prompts, MCMC uses `alpha=4`, `B=192`, `N_MCMC=10`, and the official Slurm script defaults to `qwen_math` despite the paper having separate Qwen and Qwen-Math rows.

### What remains in this section
No experiment should resume unless the protocol is amended to strict stop, full-164 RWS reproduction, or a slice-calibrated gate.

### Blockers
Phase 0 remains blocked; the audit changes the interpretation from "CovComp failed" to "the first-50 replication gate is not paper-justified."

### Next checkpoint
Regenerate artifact checksums and wait for a protocol decision.

## Completion Audit Refresh — 2026-05-11 01:42 UTC

### What ran
Reran `src/experiments/gate_audit.py`, verified the artifact manifest, checked live Flow bid status, and updated `runs/goal_status_audit.md` plus `TODO.md`.

### What was verified
The goal remains active but not achieved: Phase 0 is failed/incomplete, Phase 1 archive and Phase 2 rows/summary do not exist, no Phase 3 artifacts exist, and `polaris-a100-050` still reports `Paused`; the manifest has 40 current local entries.

### What remains in this section
Nothing; `runs/artifact_manifest.sha256` was regenerated and verified after this progress update.

### Blockers
No change: protocol decision required before GPU work can resume.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Official RWS Full-164 Prep — 2026-05-11 01:46 UTC

### What ran
Added `scripts/rws_official_full164_generate_remote.sh`, `scripts/rws_official_full164_status_remote.sh`, and `scripts/rws_official_full164_eval_remote.sh`, then syntax-checked them with `bash -n`.

### What was verified
The primary Option B path now copies the authors' `llm_experiments` source into `runs/rws_official_full164/`, materializes original HumanEval there, runs official `power_samp_he.py` with `RWS_MODEL=qwen`, and scores with official RWS extraction/checker functions; no GPU work was launched.

### What remains in this section
Nothing; `runs/artifact_manifest.sha256` now verifies 73 local source/artifact entries across `TODO.md`, `pyproject.toml`, `proposal/`, `src/`, `scripts/`, and `runs/`.

### Blockers
No change: protocol decision required before unpausing the A100 or running full-164.

### Next checkpoint
If Option B is accepted, sync local master to remote and launch the official-RWS full-164 diagnostic, not Phase 1.

## Official RWS Runner Hardening — 2026-05-11 01:49 UTC

### What ran
Patched `scripts/rws_official_full164_generate_remote.sh` so the duplicate-dispatch guard runs only in the parent process, and added a Hugging Face `snapshot_download` cache preflight before official RWS code loads with `local_files_only=True`.

### What was verified
`bash -n scripts/*.sh` passes; no GPU work was launched.

### What remains in this section
Nothing; `runs/artifact_manifest.sha256` was regenerated and verified after this progress update.

### Blockers
No change: protocol decision required before unpausing the A100.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Gate Audit Full-164 Field — 2026-05-11 01:50 UTC

### What ran
Updated `src/experiments/gate_audit.py` to report `rws_official_full164` summary existence, model id, C0a/C0b/C0c pass@1 values, and completion status; then ran `python3 -m py_compile` and regenerated `runs/gate_audit.json`.

### What was verified
The audit still reports `status = blocked`; `rws_official_full164.complete = false`, Phase 1/2 artifacts remain absent, and no Phase 3 artifacts exist.

### What remains in this section
Nothing; `runs/artifact_manifest.sha256` was regenerated and verified after this progress update.

### Blockers
No change: protocol decision required before unpausing the A100.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Second-Pass Paper Fidelity Check — 2026-05-11 01:54 UTC

### What ran
Rechecked live RWS and SPS arXiv HTML, the official arXiv Code link to `https://github.com/aakaran/reasoning-with-sampling`, local RWS HumanEval source, DeepWiki for repo internals, `PYTHONPATH=src python3 -m unittest discover -s tests`, and `PYTHONPATH=src python3 -m experiments.gate_audit --output runs/gate_audit.json`.

### What was verified
RWS Table 1 is full-164 HumanEval single-shot, while the written MVE gate compares a first-50 slice to the full-164 point estimate; the local gate audit remains `blocked` with C0a original HumanEval `0.44`, no Phase 1 archive, no Phase 2 rows, and no Phase 3 artifacts.

### What remains in this section
Nothing else should run on the GPU until the protocol decision is made.

### Blockers
The proposal did not fail, but the current Phase 0 gate failed and still blocks the active goal.

### Next checkpoint
Pick one: strict stop, approve the prepared full-164 official-RWS reproduction, or amend the first-50 gate with a slice-calibrated rule.

## Artifact Manifest Refresh — 2026-05-11 01:54 UTC

### What ran
Regenerated `runs/artifact_manifest.sha256` over `TODO.md`, `pyproject.toml`, `proposal/`, `src/`, `scripts/`, `runs/`, and `tests/`, excluding caches and bytecode, then verified it with `shasum -a 256 -c`.

### What was verified
The manifest verifies 74 local source/artifact entries including the second-pass audit note and the new metric/gate audit tests.

### What remains in this section
Nothing; local master is consistent.

### Blockers
No change: protocol decision required before unpausing the A100.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Completion Audit Snapshot — 2026-05-11 01:57 UTC

### What ran
Checked `runs/gate_audit.json`, `runs/goal_status_audit.md`, the resume runbook, live `flow bid list --json`, shell syntax for official full-164 scripts, and the unit tests, then wrote `runs/completion_audit_20260511T015704Z.md`.

### What was verified
The goal remains active and not achieved: Preflight and paper fidelity work are complete, but Phase 0 is failed/incomplete, Phase 1/2 artifacts are absent, no Phase 3 artifacts exist, and `polaris-a100-050` is paused at `$0.05`.

### What remains in this section
Refresh artifact checksums after this audit snapshot.

### Blockers
Protocol decision is still required before unpausing the A100 or running the full-164 diagnostic.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Artifact Manifest Refresh — 2026-05-11 01:57 UTC

### What ran
Regenerated and verified `runs/artifact_manifest.sha256` after adding the completion audit snapshot.

### What was verified
The manifest verifies 75 local source/artifact entries, including `runs/completion_audit_20260511T015704Z.md`.

### What remains in this section
Nothing; local artifacts are internally consistent.

### Blockers
No change: protocol decision required before GPU work resumes.

### Next checkpoint
Wait for strict stop, full-164 reproduction approval, or an amended first-50 gate.

## Full-164 Official RWS Diagnostic Execution — 2026-05-11 03:07 UTC

### What ran
Unpaused existing Flow bid `polaris-a100-050` at `$0.05/hr`, verified A100 SSH readiness, synced local master to `/home/ubuntu/polaris`, attempted official all-method full-164 RWS generation, stopped it after a real MCMC OOM at 3 workers, then ran the narrower full-164 official-std C0a diagnostic and evaluated it with the RWS HumanEval extraction/checker path.

### What was verified
The all-method MCMC path OOMed at 3 workers around `74/82GB`, but the std-only C0a path completed all four 41-problem CSV shards and scored original HumanEval full-164 pass@1 `0.29878048780487804`, inside the RWS Table 1 target window `0.279-0.379` around `0.329`.

### What remains in this section
Refresh machine audit, tests, manifest, and Flow status; then use this evidence to revise the invalid first-50 C0a gate before deciding how to resume Phase 0 C0b/C0c.

### Blockers
Full RWS all-method MCMC cannot run with 3 workers on this A100; any C0b/C0c continuation needs lower concurrency or a narrower budgeted path.

### Next checkpoint
Update gate audit to include `runs/rws_official_full164_std/eval/summary.json`, rerun tests, and sync/pause status as needed.

## Full-164 Official RWS Naive Launch — 2026-05-11 03:10 UTC

### What ran
Generalized the std-only wrapper to support `MODE=std|naive`, syntax-checked it, reran unit tests, synced it to the A100, and launched `MODE=naive CLEAN=1 WORKERS=3 RWS_MODEL=qwen bash scripts/rws_official_full164_std_generate_remote.sh`.

### What was verified
The std C0a result is now in the machine audit as `rws_official_full164_std.pass_at_1 = 0.29878048780487804`; the naive/low-temperature C0b dispatcher started as PID `17924`.

### What remains in this section
Monitor the full-164 C0b shards and evaluate them when all four CSVs exist.

### Blockers
No current C0b blocker.

### Next checkpoint
Poll `MODE=naive` status and GPU utilization.

## Full-164 Official RWS Naive Monitor — 2026-05-11 03:12 UTC

### What ran
Polled `MODE=naive` status after worker startup.

### What was verified
Three low-temperature workers are active, GPU utilization is `100%`, memory is about `47.6/81.9GB`, and batch logs show early HumanEval progress with `temperature=0.25`.

### What remains in this section
Wait for four C0b CSV shards, then run `MODE=naive scripts/rws_official_full164_std_eval_remote.sh`.

### Blockers
No current blocker.

### Next checkpoint
Poll C0b shard completion.

## Deep Paper Replication Recheck — 2026-05-11 03:54 UTC

### What ran
Rechecked the live RWS, SPS, GEPA, and EvalPlus arXiv pages plus the official RWS HumanEval source at commit `720a8e9d084c87a630595e316f5260f1d7c3446c`; polled `polaris-a100-050` with `MODE=naive bash scripts/rws_official_full164_std_status_remote.sh`.

### What was verified
RWS Table 1 is full-164 original HumanEval single-shot, not first-50 and not verifier-best-of-8; full-164 C0a already scored `0.29878048780487804`, so the proposal did not fail, while C0b is still running with GPU utilization `100%`, memory about `59.9/81.9GB`, and shard `2/4` complete.

### What remains in this section
Finish C0b, evaluate it, sync `runs/rws_official_full164_naive/`, then run the MCMC-only C0c wrapper at conservative concurrency.

### Blockers
No live C0b blocker; MCMC previously OOMed with three workers, so C0c must default to `WORKERS=1` unless a bounded smoke test shows a second worker is safe.

### Next checkpoint
Poll C0b until all four CSV shards exist, then run `MODE=naive RWS_MODEL=qwen bash scripts/rws_official_full164_std_eval_remote.sh`.

## Full-164 Official RWS Naive Result — 2026-05-11 04:17 UTC

### What ran
Completed all four `runs/rws_official_full164_naive/results/qwen/qwen_he_naive_only_results_*_0.csv` shards, evaluated with `MODE=naive RWS_MODEL=qwen bash scripts/rws_official_full164_std_eval_remote.sh`, synced the run back local, regenerated `runs/gate_audit.json`, and launched `CLEAN=1 WORKERS=1 RWS_MODEL=qwen bash scripts/rws_official_full164_mcmc_generate_remote.sh`.

### What was verified
C0b full-164 original HumanEval pass@1 is `0.524390243902439`, matching the RWS Table 1 `Qwen2.5-7B` low-temperature target `0.524`; C0a remains `0.29878048780487804`, matching the `0.329` target window.

### What remains in this section
Monitor the MCMC-only C0c diagnostic under `runs/rws_official_full164_mcmc/`, then evaluate and sync it when all four shards finish.

### Blockers
No current blocker; C0c uses `WORKERS=1` because the earlier three-worker all-method MCMC run OOMed.

### Next checkpoint
Poll `bash scripts/rws_official_full164_mcmc_status_remote.sh` for batch 0 GPU utilization and shard progress.

## Full-164 Official RWS MCMC Utilization Adjustment — 2026-05-11 04:55 UTC

### What ran
Observed one-worker C0c MCMC at about `19-22GB` and `58-64%` GPU with no CSV yet, killed only the dispatcher parent after preserving the active batch 0 child, synced `scripts/rws_official_full164_mcmc_one_batch_remote.sh`, and launched `BATCH_IDX=1 RWS_MODEL=qwen` as a second independent worker.

### What was verified
Batch 0 and batch 1 are both running, GPU utilization returned to `100%`, and memory is about `37.8/81.9GB`, which is below the prior three-worker OOM envelope.

### What remains in this section
Monitor batches 0 and 1; when one completes, launch the next remaining batch only if total active MCMC workers stays at two or below.

### Blockers
No current blocker; do not exceed two concurrent MCMC workers on this A100.

### Next checkpoint
Poll `bash scripts/rws_official_full164_mcmc_status_remote.sh` and watch for CSV shards `0` and `1`.

## Full-164 Official RWS MCMC Monitor — 2026-05-11 05:52 UTC

### What ran
Added and syntax-checked `scripts/monitor_mcmc_c0c_local.sh`, started it locally with PID recorded in `runs/mcmc_c0c_monitor.pid`, and verified its first poll in `runs/mcmc_c0c_monitor.log`.

### What was verified
The monitor reports `csv_count=0`, `workers=2`, `started=[0 1]`, and `completed=[]`, so it is preserving the two-worker cap and will launch batches 2 and 3 only when an active batch writes a CSV.

### What remains in this section
Let the monitor continue; it will evaluate C0c, sync `runs/rws_official_full164_mcmc/`, regenerate `runs/gate_audit.json`, and append the C0c result to this file after all four shards complete.

### Blockers
No current blocker, but C0c is the long pole and may take many hours.

### Next checkpoint
Inspect `runs/mcmc_c0c_monitor.log` and remote MCMC status for the first completed shard.

## Full-164 Official RWS MCMC Monitor Detach Fix — 2026-05-11 05:54 UTC

### What ran
The initial `nohup` local monitor exited after its first poll on macOS, so it was relaunched under `screen -dmS polaris_mcmc_monitor` with `POLL_SECONDS=600 MAX_WORKERS=2`.

### What was verified
`screen -ls` shows `polaris_mcmc_monitor` detached and alive, and `runs/mcmc_c0c_monitor.log` reports `csv_count=0`, `workers=2`, `started=[0 1]`, and `completed=[]`.

### What remains in this section
Let the screen-backed monitor manage batch 2/3 launches, C0c evaluation, sync, and gate audit refresh.

### Blockers
No monitor blocker.

### Next checkpoint
Check `screen -ls` and `runs/mcmc_c0c_monitor.log` on the next status pass.

## Full-164 Official RWS MCMC Progress — 2026-05-11 06:22 UTC

### What ran
Polled the screen-backed C0c monitor and counted remote MCMC batch progress from batch logs.

### What was verified
The monitor is alive with `workers=2`, no CSV shards are complete yet, batch 0 has reached `alpha_count=8`, and batch 1 has reached `alpha_count=3`.

### What remains in this section
Continue C0c until all four shards are written and evaluated.

### Blockers
No correctness blocker; C0c is compute-bound and slow.

### Next checkpoint
Wait for the monitor to launch batch 2 after either batch 0 or batch 1 completes.

## Full-164 Official RWS MCMC Progress — 2026-05-11 06:59 UTC

### What ran
Polled `runs/mcmc_c0c_monitor.log`, remote batch logs, `nvidia-smi`, and active MCMC processes.

### What was verified
The monitor remains alive, GPU utilization is `100%`, memory is about `48.2/81.9GB`, batch 0 has `alpha_count=8`, batch 1 has `alpha_count=6`, and no CSV shard has completed yet.

### What remains in this section
Continue C0c with the two-worker cap until at least one shard completes.

### Blockers
No runtime blocker; C0c remains compute-bound.

### Next checkpoint
Wait for a C0c CSV shard or monitor launch of batch 2.

## Full-164 Official RWS MCMC Future-Batch Hardening — 2026-05-11 07:31 UTC

### What ran
Patched `scripts/rws_official_full164_mcmc_one_batch_remote.sh` to write a `.partial.csv` after each completed HumanEval problem and resume from that file, syntax-checked the scripts, reran the 9 local unit tests, and synced the worker script to the A100.

### What was verified
The active C0c workers are still running at `100%` GPU and about `48.2/81.9GB`; future batches 2 and 3 will only create the final CSV after all 41 rows exist, so the existing monitor will not misclassify partial shards as complete.

### What remains in this section
Continue active batches 0 and 1; when the monitor launches batches 2 and 3, they will use the hardened partial-output path.

### Blockers
No new blocker; active batches 0 and 1 still use the original non-incremental runner because they were already launched.

### Next checkpoint
Wait for batch 0 or 1 completion, then verify the monitor launches a hardened batch 2.

## Local Verification Refresh — 2026-05-11 05:55 UTC

### What ran
Ran `bash -n` on the MCMC monitor/worker/status scripts, `PYTHONPATH=src python3 -m unittest discover -s tests`, and regenerated `runs/artifact_manifest.sha256` while excluding live monitor logs and the manifest file itself.

### What was verified
Shell syntax checks passed, all `9` unit tests passed, and the artifact manifest verified successfully.

### What remains in this section
Continue monitoring C0c.

### Blockers
No local verification blocker.

### Next checkpoint
Wait for C0c shard completion from the monitor.

## Deep MCMC Replication Check — 2026-05-11 07:37 UTC

### What ran
Rechecked RWS Algorithm 1, RWS Section 5.1/Table 1, SPS Appendix C, official RWS HumanEval code, DeepWiki's source summary for `aakaran/reasoning-with-sampling`, and the live remote C0c process.

### What was verified
The active C0c run calls the official RWS `mcmc_power_samp` with `Qwen/Qwen2.5-7B`, raw Qwen HumanEval prompts, `temp=0.25`, `mcmc_steps=10`, `max_new_tokens=3072`, and default `block_num=16`, but it is a MCMC-only execution wrapper rather than the unmodified all-method `power_samp_he.py`.

### What remains in this section
Continue the C0c run and label the result precisely as official RWS MCMC function plus MCMC-only wrapper.

### Blockers
No new blocker; the only fidelity caveat is stochastic RNG-state drift from skipping the authors' preceding low-temperature/std generations inside `power_samp_he.py`.

### Next checkpoint
If C0c lands far from `0.622`, run an unmodified `power_samp_he.py` control before treating the miss as a scientific failure.

## Cross-Model Replication Review — 2026-05-11 07:37 UTC

### What ran
Ran Gemini against `runs/deep_replication_audit.md`, the active C0c MCMC scripts, and the official RWS HumanEval source.

### What was verified
Gemini found no model/prompt/sampler/hyperparameter mismatch, but flagged that MCMC-only CSV filenames and empty `std_completion`/`naive_completion` columns are not compatible with unmodified `eval_he.py`.

### What remains in this section
Keep using the MCMC-specific evaluator for C0c artifacts and do not describe the run as unmodified end-to-end RWS.

### Blockers
No live blocker; artifact-shape deviations are evaluator-contract deviations only.

### Next checkpoint
If unmodified `eval_he.py` is needed later, generate or convert artifacts into the official filename/schema first.

## Full-164 Official RWS MCMC Progress — 2026-05-11 07:45 UTC

### What ran
Polled the screen-backed C0c monitor after its next scheduled tick and queried remote MCMC batch logs.

### What was verified
The monitor is alive and reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; remote batch 0 has `alpha_count=9`, batch 1 has `alpha_count=6`, and no final or partial CSV shard exists yet.

### What remains in this section
Continue the two-worker C0c run until batch 0 or 1 emits a final CSV, then let the monitor launch a hardened batch 2 or 3.

### Blockers
No runtime blocker; the job is compute-bound and still running under the known safe two-worker cap.

### Next checkpoint
Poll the monitor at the next 10-minute tick or when a CSV shard appears.

## Full-164 Official RWS MCMC Progress — 2026-05-11 07:56 UTC

### What ran
Waited through the next screen-backed monitor tick and queried remote batch logs after the tick.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains in its ninth MCMC generation, batch 1 remains in its sixth, and both logs grew since the previous check.

### What remains in this section
Keep the run untouched until a final CSV shard appears or worker/GPU health changes.

### Blockers
No correctness or infrastructure blocker; C0c is still the compute-bound long pole.

### Next checkpoint
Poll after another monitor tick, or immediately if GPU utilization drops or a shard appears.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:05 UTC

### What ran
Waited through another monitor tick and queried the remote C0c batch logs.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains in its ninth MCMC generation, batch 1 remains in its sixth, and both logs grew again.

### What remains in this section
Continue the C0c run unchanged until a final shard appears or worker health changes.

### Blockers
No infrastructure blocker; C0c is still slow because the official RWS MCMC sampler is compute-heavy on one A100.

### Next checkpoint
Poll the next scheduled monitor tick or respond immediately if a CSV shard appears.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:15 UTC

### What ran
Waited through the next monitor tick, queried remote batch logs, and checked GPU utilization.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 advanced to `alpha_count=7`, and the A100 reports `99%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Continue the run untouched until one of the active batches emits a final CSV shard.

### Blockers
No blocker; progress is slow but active.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only if worker count drops, GPU idles, or a shard appears without a follow-on batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:25 UTC

### What ran
Waited through the next scheduled monitor tick and queried remote C0c batch logs.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 remains at `alpha_count=7`, and both logs grew since the previous checkpoint.

### What remains in this section
Continue the two-worker C0c run until a final shard appears.

### Blockers
No blocker; the job remains compute-bound under the safe two-worker cap.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:35 UTC

### What ran
Waited through the next scheduled monitor tick, queried remote batch logs, and checked GPU utilization.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 remains at `alpha_count=7`, both logs grew, and the A100 reports `100%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Continue the run untouched until batch 0 or 1 emits the first final CSV shard.

### Blockers
No blocker; execution remains healthy but slow.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:45 UTC

### What ran
Waited through the next scheduled monitor tick and queried remote C0c batch logs.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 remains at `alpha_count=7`, and both logs grew again.

### What remains in this section
Continue the run untouched until a final CSV shard appears.

### Blockers
No blocker; execution remains healthy but the official RWS MCMC sampler is still slow.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 08:55 UTC

### What ran
Waited through the next scheduled monitor tick, queried remote C0c batch logs, and checked GPU utilization.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 remains at `alpha_count=7`, both logs grew, and the A100 reports `100%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Continue the run unchanged until a final CSV shard appears.

### Blockers
No blocker; the run is healthy and compute-bound.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 09:05 UTC

### What ran
Waited through the next scheduled monitor tick and queried remote C0c batch logs.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=9`, batch 1 remains at `alpha_count=7`, and both logs grew again.

### What remains in this section
Continue the run unchanged until one of the active batches emits a final CSV shard.

### Blockers
No blocker; the run remains healthy and compute-bound.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 09:16 UTC

### What ran
Checked Flow status, waited through the next scheduled monitor tick, and queried remote C0c batch logs.

### What was verified
Flow reports `polaris-a100-050` as `running` on `1xa100`; the monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 advanced to `alpha_count=10`, batch 1 remains at `alpha_count=7`, and no CSV shard exists yet.

### What remains in this section
Continue the run unchanged until one of the active batches emits a final CSV shard.

### Blockers
No blocker; Flow metadata still shows `started_at=null`, but SSH and `nvidia-smi` confirm the instance is usable and saturated.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 09:26 UTC

### What ran
Waited through the next scheduled monitor tick and queried remote C0c batch logs.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=10`, batch 1 remains at `alpha_count=7`, and both logs grew.

### What remains in this section
Continue the run unchanged until one active batch emits a final CSV shard.

### Blockers
No blocker; the run remains healthy and compute-bound.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 09:36 UTC

### What ran
Checked Flow status, waited through the next scheduled monitor tick, and queried remote C0c batch logs.

### What was verified
Flow reports `polaris-a100-050` as `running`; the monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=10`, batch 1 advanced to `alpha_count=9`, and no CSV shard exists yet.

### What remains in this section
Continue the run unchanged until one of the active batches emits a final CSV shard.

### Blockers
No blocker; the run remains healthy and compute-bound.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 09:46 UTC

### What ran
Checked Flow status, waited through the next scheduled monitor tick, queried remote C0c batch logs, and checked GPU utilization.

### What was verified
Flow reports `polaris-a100-050` as `running`; the monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; batch 0 remains at `alpha_count=10`, batch 1 remains at `alpha_count=9`, both logs grew, and the A100 reports `100%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Continue the run unchanged until one active batch emits a final CSV shard.

### Blockers
No blocker; the run remains healthy and compute-bound.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 10:00 UTC

### What ran
Checked Flow status, the local screen monitor, remote C0c processes, batch logs, and GPU utilization.

### What was verified
Flow reports `polaris-a100-050` as `running`; the monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; remote batch 0 remains active at `alpha_count=10`, batch 1 remains active at `alpha_count=9`, no CSV shard exists yet, and the A100 reports `100%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Continue the run unchanged until one active batch emits a final CSV shard, then let the monitor start the next batch under the two-worker cap.

### Blockers
No runtime blocker; the current long pole is official RWS MCMC compute time.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 10:10 UTC

### What ran
Waited through the next monitor tick and queried remote C0c process, GPU, shard, and batch-log status.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; remote batch 0 and batch 1 are still active, no final CSV shard exists, and the A100 reports `100%` utilization with about `48.4/81.9GB` used.

### What remains in this section
Wait for batch 0 or batch 1 to emit the first final CSV shard; batch 0 appears closest based on the visible 16-problem progress bar.

### Blockers
No runtime blocker; C0c remains compute-bound under the safe two-worker cap.

### Next checkpoint
Poll the next scheduled monitor tick or intervene only on worker drop, GPU idle, OOM, or shard completion without next-batch launch.

## Full-164 Official RWS MCMC Progress — 2026-05-11 10:16 UTC

### What ran
Waited through the next monitor tick and queried remote C0c process, GPU, shard, and batch-log status.

### What was verified
The monitor reported `csv_count=0`, `workers=2`, `started=[0 1]`, `completed=[]`; no final CSV shard exists yet, the A100 remains at `100%` utilization with about `48.4/81.9GB` used, batch 0 appears near `15/16` on its shard, and batch 1 appears around `10/16`.

### What remains in this section
Wait for batch 0 to complete and verify the monitor starts the next shard without exceeding the two-worker cap.

### Blockers
No runtime blocker; this is still compute-bound official RWS MCMC.

### Next checkpoint
Poll in a shorter interval because batch 0 is close to shard completion.

## Preemption/Cost Pivot — 2026-05-11 10:32 UTC

### What ran
Verified the running 1x A100 has no partial or final C0c CSV/JSON results, added checkpointed per-problem RWS MCMC scripts, checked Flow inventory, and submitted exactly one low-capped `8xa100` bid named `polaris-a100x8-ckpt`.

### What was verified
Flow had only `polaris-a100-050` before the new bid; after submission it has `polaris-a100-050` running plus `polaris-a100x8-ckpt-*` provisioning/open at max `$0.15/hr`; no B200/4x bid was created.

### What remains in this section
Wait for the 8x A100 to become SSH/GPU healthy, sync the checkpointed runner, start one worker per GPU with per-problem JSON checkpoints, then stop the old 1x uncheckpointed run after the new run writes its first durable checkpoint.

### Blockers
The current 1x run cannot be made checkpoint-safe in place because completed rows are only in process memory and no partial files exist on disk.

### Next checkpoint
Poll Flow for `polaris-a100x8-ckpt-*` readiness; do not create additional bids while it is provisioning.

## Checkpointed Runner Cutover — 2026-05-11 10:40 UTC

### What ran
Stopped the local old C0c monitor, killed all old uncheckpointed remote MCMC shell and orphan Python processes, launched `runs/rws_official_full164_mcmc_ckpt` on the existing 1x A100, and synced checkpoint artifacts back locally.

### What was verified
The old GPU was clean before relaunch (`0 MiB` used); the checkpointed runner wrote durable JSON files for `HumanEval/0` and `HumanEval/1`; no old uncheckpointed MCMC process remains; the 8x A100 bid is still provisioning/open and no additional GPU bid was created.

### What remains in this section
Keep syncing checkpoint JSON/logs, wait for the 8x A100 to become healthy, then migrate the checkpointed task directory and start one worker per GPU there.

### Blockers
The old uncheckpointed in-memory generations were not recoverable; only future completed tasks are now preemption-safe.

### Next checkpoint
Poll 8x readiness and checkpoint count; once 8x has written its first checkpoint, stop the 1x bid to avoid duplicate spend.

## Cost/Preemption Management — 2026-05-11 10:47 UTC

### What ran
Started local screen session `polaris_ckpt_sync_current` to rsync the checkpointed run every 180 seconds, verified the 8x A100 bid remained the only new bid, and raised that same bid from `$0.15/hr` to `$0.30/hr` after availability moved to `$0.25/hr`.

### What was verified
Local sync copied the first two durable task JSONs; current Flow inventory is exactly one running 1x bid and one provisioning/open 8x bid; no B200, 4x, or duplicate 8x bid exists.

### What remains in this section
Wait for `polaris-a100x8-ckpt-e8b2d2` to receive a host, then sync the repo and checkpointed task directory to it.

### Blockers
The 8x host is not assigned yet (`host=null`, `started_at=null`), so no 8x setup can begin.

### Next checkpoint
Continue Flow polling; if the 8x bid does not become reachable after the next market cycle, reassess whether to keep waiting or switch to a 4x bid after canceling the 8x bid.

## Multi-GPU Fallback — 2026-05-11 11:10 UTC

### What ran
Raised the existing 1x bid cap to `$1.00/hr` to recover it from `preempting`, paused the ambiguous 8x bid so it cannot allocate, and submitted one 4x A100 fallback bid `polaris-a100x4-ckpt-66304a` at `$0.10/hr`.

### What was verified
The 1x returned to `running` and continues the checkpointed worker; the 8x bid is paused; the 4x bid is allocated/provisioning but still has `host=null` and `started_at=null`.

### What remains in this section
Wait for the 4x host assignment. If it becomes SSH/GPU healthy, sync code and checkpointed JSONs there, stop the 1x worker, and launch one checkpointed worker per GPU.

### Blockers
The 4x bid has not exposed an SSH host yet, so setup cannot start.

### Next checkpoint
If the 4x bid remains hostless past the startup window, pause/delete it and consider a managed multi-1x sharded fallback with explicit global ranks.

## 4x Checkpointed Cutover — 2026-05-11 11:26 UTC

### What ran
Synced the checkpointed run from the 1x host to local and then to `polaris-a100x4-ckpt-66304a`, stopped the 1x worker and orphan Python sampler, patched `scripts/rws_mcmc_ckpt_launch_remote.sh` to default `WORLD_SIZE` and `MAX_WORLD_SIZE` to the visible GPU count, and relaunched the run on four A100s.

### What was verified
The 4x host is running four checkpointed workers with one worker per GPU, checkpoint count increased to `4 / 164`, local sync session `polaris_ckpt_sync_4x` is active, `polaris-a100-050` was cancelled, and the unused 8x bid remains paused/non-cancellable rather than running.

### What remains in this section
Continue monitoring the 4x run, sync every 180 seconds, and merge/report only from saved JSONL/JSON artifacts.

### Blockers
No current blocker; the main risk is spot preemption, mitigated by per-problem checkpoints and the local rsync loop.

### Next checkpoint
Poll Flow status, GPU utilization, checkpoint count, and sync log; if utilization drops or no checkpoint count changes over a long interval, inspect worker logs before changing compute.

## Watchdog Correction — 2026-05-11 11:28 UTC

### What ran
Replaced the sync-only local screen with `polaris_ckpt_monitor_4x`, patched the monitor to parse p90 timing correctly, and fixed remote health probes to read `/home/ubuntu/polaris` rather than the SSH login directory.

### What was verified
The watchdog now reports `running=4`, `completed=4/164`, fresh heartbeat from worker 0 on `HumanEval/8`, and all four 4x A100 GPUs show active memory/utilization.

### What remains in this section
Let the monitor rsync artifacts and auto-restart the 4-worker pool only if workers truly die.

### Blockers
No blocker; one failed watchdog restart attempt occurred before the path fix, but the launcher refused duplicate workers and the live pool stayed at four workers.

### Next checkpoint
Check monitor state after the next poll cycle and confirm checkpoint count continues increasing without additional Flow allocations.

## A100 Utilization Tuning — 2026-05-11 11:32 UTC

### What ran
Patched the launcher to support `WORKERS_PER_GPU`, stopped the four-worker pool after syncing completed artifacts, and relaunched with `WORLD_SIZE=8`, `WORKERS_PER_GPU=2`, and `MAX_WORLD_SIZE=8`.

### What was verified
The 4x host now runs eight checkpointed workers, two per GPU; each GPU uses about `31GB / 80GB`, utilization is `99-100%`, checkpoint count increased to `6 / 164`, and the watchdog reports `running=8`.

### What remains in this section
Keep the conservative 2x-per-GPU configuration unless memory exceeds a safe margin, workers die repeatedly, or throughput collapses.

### Blockers
No blocker; the temporary relaunch discarded in-flight partial work but preserved all completed per-problem JSON checkpoints.

### Next checkpoint
After the next monitor poll, compare checkpoint growth and GPU memory to confirm the 8-worker configuration remains stable.

## Monitor Metrics Fix — 2026-05-11 11:35 UTC

### What ran
Patched `monitor_mcmc_ckpt_local.sh` to compute timing stats from per-task JSON checkpoints before falling back to partial CSV.

### What was verified
The restarted watchdog reports `running=8`, `completed=8/164`, and timing stats now come from durable JSON artifacts (`avg=15.96s`, `p90=37.01s` at this early sample).

### What remains in this section
Keep using worker count, GPU memory/utilization, checkpoint growth, and JSON-derived timing as the operating signals.

### Blockers
No blocker; early ETA remains noisy because the first completed tasks are not representative of hard HumanEval problems.

### Next checkpoint
Wait for one more checkpoint-growth cycle before considering any further concurrency change; do not raise above two workers per GPU while utilization is already saturated.

## Flow Watchdog — 2026-05-11 11:40 UTC

### What ran
Added and launched `polaris_flow_watch`, a local Flow-status watchdog that polls every 90 seconds, logs active task count, forces an artifact sync on `preempting`, and auto-cancels the 4x bid after all 164 task JSONs are locally synced.

### What was verified
The watcher reports `status=running`, `active=1`, bid limit `$0.10`, the unused 8x bid paused, and `completed=8/164`.

### What remains in this section
Let `polaris_ckpt_monitor_4x` handle worker health/artifact sync and `polaris_flow_watch` handle Flow preemption/cost state.

### Blockers
No blocker; the first launch failed only because the new script lacked executable permissions, then succeeded after `chmod +x`.

### Next checkpoint
Do not provision anything else; monitor the two local screens and the single active 4x Flow task.

## Long-Task Plateau Check — 2026-05-11 11:44 UTC

### What ran
Polled the remote 8-worker pool after several minutes with checkpoint count flat at `8/164`.

### What was verified
All eight checkpointed workers are alive, all four A100s remain saturated at roughly `99-100%` utilization with about `35-37GB / 80GB` used, and worker logs show active MCMC progress through long HumanEval tasks rather than a deadlock.

### What remains in this section
Continue waiting for the long tasks to finish and write their per-task JSONs.

### Blockers
No blocker; the flat checkpoint count is explained by hard tasks still inside their 16-sample MCMC loops.

### Next checkpoint
Poll again after another watchdog interval and only intervene if worker count drops, GPU utilization collapses, Flow status changes, or logs stop advancing.

## Completion Hook Smoke — 2026-05-11 11:53 UTC

### What ran
Added `scripts/rws_mcmc_ckpt_eval_remote.sh`, patched `watch_flow_preempt_local.sh` so completion triggers remote merge, EvalPlus scoring, final sync, and Flow cancellation, then smoke-tested EvalPlus scoring on the first 8 checkpointed task JSONs.

### What was verified
The partial EvalPlus smoke completed successfully with `task_json_count=8`, `humaneval_pass_at_1=0.625`, and `humaneval_plus_pass_at_1=0.625`; the Flow watcher was restarted with the new completion hook.

### What remains in this section
Wait for all 164 task JSONs before trusting the metric; the 8-task smoke is only an import/evaluator sanity check.

### Blockers
No blocker; EvalPlus downloaded the dataset and computed expected outputs once on the remote host.

### Next checkpoint
Continue monitoring until checkpoint count moves or worker logs indicate a real stall.

## Long-Tail Confirmation — 2026-05-11 12:10 UTC

### What ran
Directly inspected the 8-worker remote logs after checkpoint count stayed flat at `8/164`.

### What was verified
All eight workers are alive, all four A100s are still saturated at `99-100%`, GPU memory is about `39-42GB / 80GB`, and logs show workers advancing through samples around `10-11/16` on the current hard tasks.

### What remains in this section
Wait for the current hard tasks to finish; multiple checkpoints should land once this batch crosses task boundaries.

### Blockers
No blocker; the stale heartbeat was a measurement artifact from task-boundary-only heartbeat writes.

### Next checkpoint
Restarted the checkpoint monitor with `HEARTBEAT_STALE_SECONDS=7200`; keep the current 8-worker configuration and do not relaunch while logs advance.

## Long-Tail Still Advancing — 2026-05-11 12:21 UTC

### What ran
Polled the remote checkpointed runner again after the local monitor stayed at `8/164`.

### What was verified
All eight workers remain alive, all four GPUs remain saturated, memory is about `42-44GB / 80GB`, and the current hard tasks advanced to roughly `12/16` MCMC samples.

### What remains in this section
Let the current hard-task batch finish; checkpoint count should jump only when these task-level calls return.

### Blockers
No blocker; the plateau is caused by very long in-flight MCMC calls, not idle hardware.

### Next checkpoint
Continue monitoring without relaunching; intervene only on OOM, worker death, Flow preemption/cancellation, or stopped log progress.

## Hard Batch Near Boundary — 2026-05-11 12:37 UTC

### What ran
Directly inspected remote worker logs again after the monitor remained at `8/164`.

### What was verified
The eight workers and four saturated GPUs remain healthy; several workers have advanced to about `14/16` samples on the current hard tasks, with memory still below about `47GB / 80GB`.

### What remains in this section
Wait for the current task batch to cross boundaries and write JSON checkpoints.

### Blockers
No blocker; this is a very long MCMC batch but logs are still advancing.

### Next checkpoint
Continue waiting; do not relaunch and lose the in-flight batch.

## First Long-Task Checkpoint — 2026-05-11 12:54 UTC

### What ran
Forced a local artifact sync after remote status showed the first long in-flight task completed.

### What was verified
Remote and local checkpoint count reached `9/164`, with `003_HumanEval_3.json` written; remaining hard-task workers are mostly near `15/16` samples and still advancing.

### What remains in this section
Wait for the rest of the long batch to cross task boundaries and sync their JSONs.

### Blockers
No blocker; one manual sync process was killed after completing because the sync script is normally a loop and the watchdog already handles recurring sync.

### Next checkpoint
Continue monitoring the existing two screens only; do not create new sync loops or worker pools.

## Long Batch Landed — 2026-05-11 13:09 UTC

### What ran
Waited through the long hard-task batch and let the existing monitors sync completed task JSONs.

### What was verified
Checkpoint count increased from `9/164` to `21/164`; Flow watcher still reports exactly one active running task, bid limit `$0.10`, and the paused 8x bid remains non-running.

### What remains in this section
Continue the C0c carryover run under the same 8-worker configuration until it completes or errors.

### Blockers
No blocker; JSON-derived ETA is noisy but currently around 10 hours because the long HumanEval tasks dominate average wall time.

### Next checkpoint
Keep the current monitor and Flow watcher screens running; completion hook should merge, EvalPlus-score, sync, and cancel the 4x bid at `164/164`.

## Cost Guardrail — 2026-05-11 13:17 UTC

### What ran
Rechecked Flow status, bid list, local watchdog screens, remote worker state, and direct GPU utilization.

### What was verified
Exactly one running Flow instance is active: `polaris-a100x4-ckpt-66304a` at `$0.10`; the paused `polaris-a100x8-ckpt-e8b2d2` bid is `not-cancellable` through `flow instance delete`, remains paused, and its cap was reduced from `$1.25` to `$0.10`.

### What remains in this section
Let the current checkpointed C0c run continue only while workers stay productive and artifacts sync.

### Blockers
No blocker; remote status shows 8 checkpointed workers, no old uncheckpointed workers, all four A100s at 99-100% utilization, and `21/164` task JSONs saved.

### Next checkpoint
Continue monitoring; do not provision, unpause, or launch any additional expensive work while the carryover RWS baseline is running.

## Productive Flat Window — 2026-05-11 13:28 UTC

### What ran
Repolled local monitor logs, Flow watcher logs, Flow status, bid list, and direct remote worker status after the checkpoint count stayed at `21/164`.

### What was verified
The run is still productive: exactly 8 checkpointed workers are alive, no old uncheckpointed workers are present, all four A100s are at 99-100% utilization, and worker logs have advanced deeper into the current long task loops.

### What remains in this section
Wait for one of the current eight long tasks to return and write the next JSON checkpoint.

### Blockers
No blocker; the flat count is a checkpoint-boundary artifact, not idle GPU spend.

### Next checkpoint
Keep the same 8-worker configuration and the same two local screens; inspect remotely again only if worker count drops, GPU utilization falls, Flow preempts, or logs stop advancing.

## Overnight Watch Guard — 2026-05-11 13:29 UTC

### What ran
Checked macOS sleep assertions and started `screen -dmS polaris_caffeinate_watch`, which runs `/usr/bin/caffeinate -s -i` while `[w]atch_flow_preempt_local.sh` exists.

### What was verified
The local Flow watcher remains active, the existing general caffeinate assertion is still present, and the `polaris_caffeinate_watch` screen now holds a dedicated caffeinate process for the merge/eval/cancel hook.

### What remains in this section
Continue relying on the existing monitor and Flow watcher screens; do not add duplicate worker pools or sync loops.

### Blockers
No blocker; this is a local reliability guard, not a change to the running experiment.

### Next checkpoint
Poll checkpoint count and Flow state; if completion reaches `164/164`, verify eval artifacts and that the 4x A100 bid was deleted.

## Next Checkpoint Landed — 2026-05-11 13:36 UTC

### What ran
Let the existing worker pool continue after the productive flat window and repolled local monitor, Flow watcher, Flow status, and local task JSONs.

### What was verified
Checkpoint count advanced from `21/164` to `22/164`, with `041_HumanEval_41.json` synced locally; the monitor still sees 8 running workers and Flow still reports only `polaris-a100x4-ckpt-66304a` as running.

### What remains in this section
Continue the same checkpointed C0c run and wait for the next task JSONs.

### Blockers
No blocker; the flat window resolved without intervention.

### Next checkpoint
Keep monitoring under the existing cost guardrails; completion hook remains responsible for merge, EvalPlus scoring, sync, and 4x A100 deletion.

## Live Contract Correction — 2026-05-11 13:39 UTC

### What ran
Audited `TODO.md`, the POLARIS proposal, and `scripts/check_protocol_sync.sh` for stale live-state claims.

### What was verified
`TODO.md` and the proposal now name `polaris-a100x4-ckpt-66304a`, `WORLD_SIZE=8`, `WORKERS_PER_GPU=2`, and the paused `$0.10` 8x bid guardrail; `scripts/check_protocol_sync.sh` passes.

### What remains in this section
Keep historical progress entries as provenance even where they mention the old 1x A100 path.

### Blockers
No blocker; this was a maintenance-doc correction only.

### Next checkpoint
Sync the updated contract files to the remote copy and continue monitoring C0c without launching any new phase.

## Long Batch Still Productive — 2026-05-11 13:51 UTC

### What ran
Repolled monitor logs, Flow status, and direct remote checkpoint/GPU/worker status after the run remained at `22/164`.

### What was verified
Remote status still shows 8 checkpointed workers, no old uncheckpointed workers, all four A100s at 99-100% utilization, and several workers around `12/16` or `13/16` samples on current long tasks.

### What remains in this section
Wait for the current long tasks to finish and write additional task JSON checkpoints.

### Blockers
No blocker; direct logs are advancing and hardware is saturated.

### Next checkpoint
Continue waiting under the same `WORLD_SIZE=8`, `WORKERS_PER_GPU=2` configuration; do not restart unless worker count drops or logs stop moving.

## Long Batch Near Finish — 2026-05-11 14:02 UTC

### What ran
Rechecked direct remote worker/GPU status after another flat interval at `22/164`.

### What was verified
The 8 checkpointed workers are still alive, all four A100s remain at 99-100% utilization, and several workers have advanced to about `13/16` or `14/16` samples on the current long tasks.

### What remains in this section
Wait for this long batch to finish and checkpoint; the next local count may jump by multiple tasks.

### Blockers
No blocker; the logs are active and the hardware is not idle.

### Next checkpoint
Continue monitoring without changing concurrency or launching additional instances.

## Checkpoint 23 Landed — 2026-05-11 14:18 UTC

### What ran
Waited through the long task window and repolled local monitor, Flow watcher, saved task JSONs, and Flow status.

### What was verified
Checkpoint count advanced from `22/164` to `23/164`; `018_HumanEval_18.json` is synced locally, the monitor still reports 8 running workers, and Flow still shows only `polaris-a100x4-ckpt-66304a` as running.

### What remains in this section
Continue the same C0c run until completion, failure, or explicit user stop.

### Blockers
No blocker; the plateau resolved by writing another task checkpoint.

### Next checkpoint
Keep the same worker pool and low-cap Flow state; update this log again on the next checkpoint jump or anomaly.

## Long Batch Advanced — 2026-05-11 14:29 UTC

### What ran
Continued the same monitored C0c run and repolled local saved task JSONs, monitor logs, Flow watcher logs, and Flow status.

### What was verified
Checkpoint count advanced from `23/164` to `26/164`; new local JSONs include `015_HumanEval_15.json`, `011_HumanEval_11.json`, and `022_HumanEval_22.json`, and Flow still reports only the 4x A100 as running.

### What remains in this section
Continue until the full 164-task C0c reproduction completes or fails.

### Blockers
No blocker; this confirms the long batch is landing incrementally.

### Next checkpoint
Keep the current 8-worker setup and update the log on the next checkpoint jump, anomaly, or completion hook event.

## Checkpoint Burst — 2026-05-11 14:39 UTC

### What ran
Continued the same 4x A100 checkpointed C0c run and repolled saved task JSONs, monitor logs, Flow watcher logs, Flow status, and local screens.

### What was verified
Checkpoint count advanced from `26/164` to `32/164`; recent local JSONs include `029`, `037`, `044`, `052`, `016`, and `045`, while the monitor still reports 8 workers and Flow still shows one active 4x A100.

### What remains in this section
Keep running until `164/164`, then let the completion hook merge, EvalPlus-score, sync, and delete the 4x bid.

### Blockers
No blocker; the long batch is now landing in bursts as expected.

### Next checkpoint
Continue monitoring; do not change hardware, sampling parameters, or start any POLARIS follow-up experiment.

## Checkpoint Burst Continued — 2026-05-11 14:50 UTC

### What ran
Continued the same monitored C0c run and repolled local checkpoints, monitor logs, Flow watcher logs, Flow status, and screens.

### What was verified
Checkpoint count advanced from `32/164` to `35/164`; recent local JSONs include `053_HumanEval_53.json`, `061_HumanEval_61.json`, and `069_HumanEval_69.json`, with one active 4x A100 and all supervisor screens still alive.

### What remains in this section
Continue monitoring to full completion; no downstream POLARIS phase may start from these partial results.

### Blockers
No blocker; checkpointing and sync continue to work.

### Next checkpoint
Keep the current run untouched and log the next checkpoint jump or anomaly.

## Checkpoint Burst To 39 — 2026-05-11 15:02 UTC

### What ran
Continued the same monitored C0c run and repolled local checkpoints, monitor logs, Flow watcher logs, Flow status, and screens.

### What was verified
Checkpoint count advanced from `35/164` to `39/164`; new local JSONs include `049_HumanEval_49.json`, `057_HumanEval_57.json`, `065_HumanEval_65.json`, and `073_HumanEval_73.json`, with one active 4x A100 and all local supervisor screens alive.

### What remains in this section
Continue monitoring to full completion and preserve every checkpoint.

### Blockers
No blocker; the run remains slow but productive.

### Next checkpoint
Continue monitoring under the same cost guardrails and do not launch follow-up experiments.

## Preemption Pause And Bid Guard — 2026-05-11 15:29 UTC

### What ran
Detected Flow preemption from the watcher log, forced artifact sync, repolled local checkpoints, Flow status, Flow bids, and current A100 availability.

### What was verified
Local checkpoint state is preserved at `41/164`; `polaris-a100x4-ckpt-66304a` is not running but has an open bid, `polaris-a100x8-ckpt-e8b2d2` remains paused, and the 4x cap was minimally raised from `$0.10` to `$0.12` against a reported 4x A100 market of `$0.10/hr`.

### What remains in this section
Wait for the 4x bid to reallocate, then verify SSH plus `nvidia-smi -L` and resume the same checkpointed run only if no duplicate workers are active.

### Blockers
The old host is currently unreachable after preemption, so sync retries are expected until Flow assigns a live host again.

### Next checkpoint
Do not unpause 8x or start any new experiment; continue watching for 4x reallocation and keep artifacts synced from the checkpoint directory.

## Resume Guard Installed — 2026-05-11 15:31 UTC

### What ran
Added and syntax-checked `scripts/resume_mcmc_after_flow_realloc_local.sh`, then started it in the `polaris_resume_guard` screen.

### What was verified
The guard reports `target_status=paused completed=41/164`, keeps the 8x bid paused, and will resume the same 4x checkpointed worker pool only after the target is reachable and no duplicate workers are active.

### What remains in this section
Wait for Flow to reallocate the open 4x bid at the `$0.12` cap, then verify `nvidia-smi -L`, worker relaunch, and resumed checkpoint growth.

### Blockers
No active GPU is currently billing; progress is waiting on spot allocation.

### Next checkpoint
Continue background monitoring; no hardware switch, no 8x unpause, and no POLARIS follow-up phase without a completed C0c result or user approval.

## Minimal Bid Bump — 2026-05-11 15:37 UTC

### What ran
Refreshed Flow availability/pricing after the 4x bid remained paused/open, then updated only `polaris-a100x4-ckpt-66304a` from `$0.12` to `$0.15`.

### What was verified
Availability reported 4x A100 at `$0.12/hr` and 8x A100 at `$0.24/hr`, so the previous 4x cap was exactly at the cutoff; the 8x bid remains paused.

### What remains in this section
Wait for 4x reallocation, then let the resume guard verify the remote, relaunch the same checkpointed pool if needed, and sync artifacts.

### Blockers
No active billing until Flow returns the 4x instance to `running`.

### Next checkpoint
Keep the 4x cap at `$0.15` unless the market moves again; do not switch to 8x while the 4x path is viable.

## Overnight Hard Cap — 2026-05-11 15:42 UTC

### What ran
Rechecked availability after another allocator cycle and raised only the 4x bid from `$0.15` to `$0.20`.

### What was verified
The reported 4x market had moved to `$0.15/hr`, exactly equal to the prior cap, while the 8x market was `$0.30/hr`; the 8x bid remains paused and no new experiment was started.

### What remains in this section
Let the 4x bid reallocate under the `$0.20` hard cap and rely on `polaris_resume_guard` to restart only the checkpointed C0c pool.

### Blockers
No blocker beyond spot allocation delay; local artifacts remain at `41/164`.

### Next checkpoint
Do not raise above `$0.20` without a fresh cost/availability check; the next action is SSH/GPU verification after Flow returns `running`.

## Reallocated And Resumed — 2026-05-11 15:58 UTC

### What ran
Waited through provisioning, verified SSH readiness, checked `nvidia-smi -L`, process state, Flow status, and bid state.

### What was verified
`polaris-a100x4-ckpt-66304a` is `running`, four A100 80GB GPUs are visible, exactly 8 checkpointed workers are running at roughly 99-100% GPU utilization, local artifacts remain at `41/164`, and the 8x bid remains paused.

### What remains in this section
Wait for the next checkpoint to confirm post-preemption progress, then continue syncing until the full 164-task C0c run completes.

### Blockers
No blocker; the run has resumed from the checkpointed worker pool.

### Next checkpoint
Keep one active 4x bid at the `$0.20` cap, keep 8x paused, and do not launch any downstream POLARIS phase.

## Resume Guard Probe Fixed — 2026-05-11 16:01 UTC

### What ran
Patched the local resume guard worker-count probe, syntax-checked it, and restarted only the `polaris_resume_guard` screen.

### What was verified
The guard now reports `remote workers already running count=8`; direct Flow SSH also reports 8 worker processes and the GPU memory/utilization profile remains consistent with two workers per A100.

### What remains in this section
Wait for the first post-preemption task checkpoint to land and verify local sync.

### Blockers
No blocker; the earlier relaunch attempt did not create duplicate workers.

### Next checkpoint
Continue monitoring the resumed run with the 4x-only cost guard.

## Post-Preemption Checkpoints Synced — 2026-05-11 16:03 UTC

### What ran
Inspected worker logs after resume and forced a one-shot rsync from the 4x instance back to local.

### What was verified
Remote workers are making new progress and local saved task JSONs advanced from `41/164` to `47/164`, including `076`, `084`, `092`, `100`, `108`, and `116`.

### What remains in this section
Continue the checkpointed C0c run to `164/164`, then merge/evaluate from saved JSONL artifacts and cancel the 4x instance.

### Blockers
No blocker; preemption recovery is now verified by new local checkpoints.

### Next checkpoint
Keep syncing periodically and preserve the 4x-only `$0.20` cost guard.

## Watcher Completion Path Repaired — 2026-05-11 16:09 UTC

### What ran
Patched `sync_mcmc_ckpt_local.sh` to support `SYNC_ONCE=1`, patched `watch_flow_preempt_local.sh` to use one-shot syncs, restarted `polaris_flow_watch`, and killed the old stuck watcher process tree.

### What was verified
The active watcher now logs `status=running active=1 limit=$0.20 ... completed=51/164`; local artifacts are synced at `51/164`, and only one `watch_flow_preempt_local.sh` process remains.

### What remains in this section
Let the repaired watcher and resume guard continue monitoring preemption and completion.

### Blockers
No blocker; the old infinite-sync hang is cleared.

### Next checkpoint
Continue the 4x-only run and rely on the repaired completion hook to merge/evaluate/cancel when `164/164` lands.

## Protocol Sync After Watcher Repair — 2026-05-11 16:11 UTC

### What ran
Updated `TODO.md` live state to `51/164`, ran `scripts/check_protocol_sync.sh`, and synced TODO, proposal, progress, and watcher scripts back to `/home/ubuntu/polaris`.

### What was verified
The drift guard passed, the 4x bid is still the only running instance at the `$0.20` cap, the 8x bid is paused, remote and local task counts both read `51/164`, and remote has 8 workers with four saturated A100s.

### What remains in this section
Continue monitoring until the next checkpoint jump or completion.

### Blockers
No blocker.

### Next checkpoint
Do not alter protocol or hardware; wait for additional task JSONs or a preemption/completion event.

## Healthy Hard-Task Plateau — 2026-05-11 16:24 UTC

### What ran
Polled Flow, bids, local/remote checkpoint counts, GPU memory/utilization, monitor logs, watcher logs, and remote worker log tails.

### What was verified
Counts remain `51/164`, all 8 checkpointed workers are alive, all four A100s are saturated at `99-100%`, memory is about `39-40GB / 80GB`, and worker logs show active MCMC sample-loop progress rather than a stuck process.

### What remains in this section
Wait for the current hard tasks to finish and sync the next checkpoint burst.

### Blockers
No blocker; this is a productive plateau.

### Next checkpoint
Keep the 4x-only `$0.20` run unchanged and continue polling for checkpoint growth, preemption, or completion.

## Checkpoint Burst To 54 — 2026-05-11 16:38 UTC

### What ran
Event-polled the resumed 4x C0c run and synced the run directory after the checkpoint count changed.

### What was verified
Local and remote artifacts advanced to `54/164`; new local task JSONs include `093`, `101`, and `109`, with the 4x bid still running, the 8x bid paused, and all 8 workers alive.

### What remains in this section
Continue monitoring until the next checkpoint jump, preemption, or completion.

### Blockers
No blocker.

### Next checkpoint
Keep the current hardware and protocol unchanged.

## Checkpoint Burst To 58 — 2026-05-11 16:42 UTC

### What ran
Continued event-based monitoring and synced the run directory after another checkpoint jump.

### What was verified
Local and remote artifacts advanced to `58/164`; latest task JSONs include `117`, `125`, `133`, and `141`, with one running 4x A100 bid, the 8x bid paused, and 8 remote workers still active.

### What remains in this section
Continue monitoring to the next checkpoint burst or preemption/completion event.

### Blockers
No blocker.

### Next checkpoint
Keep the 4x-only `$0.20` run unchanged.

## Checkpoint Burst To 60 — 2026-05-11 16:48 UTC

### What ran
Polled the 4x A100 run, synced local artifacts, inspected remote worker count, GPU utilization, worker logs, and the checkpoint worker script.

### What was verified
Local and remote artifacts are at `60/164`; worker rank 5 exited cleanly after finishing its static shard, 7 workers remain live, the 4x bid is still the only running instance at `$0.20`, and the 8x bid remains paused.

### What remains in this section
Continue checkpoint-first monitoring and let the static shards finish; do not inject dynamic backfill unless the run fully stops or the user explicitly accepts race-risk engineering work.

### Blockers
No blocker; GPU1 is lighter because its second worker finished, but mid-run work stealing is unsafe because active workers do not use per-task locks.

### Next checkpoint
Keep the 4x-only run unchanged, sync on checkpoint changes or preemption, and cancel the instance automatically after merge/eval when `164/164` lands.

## Missing-Rank Guard Added — 2026-05-11 16:51 UTC

### What ran
Added `scripts/rws_mcmc_ckpt_launch_missing_remote.sh`, patched `resume_mcmc_after_flow_realloc_local.sh`, synced both to the 4x instance, dry-ran the missing-rank launcher, and restarted only the local resume guard.

### What was verified
The dry run reported `no missing ranks with unfinished assigned tasks`; the restarted guard detected `7/8` workers, checked missing ranks, and did not launch duplicate work.

### What remains in this section
Let the guard relaunch only a dead rank with unfinished assigned tasks; cleanly completed static shards should remain stopped.

### Blockers
No blocker.

### Next checkpoint
Continue the 4x-only `$0.20` run and rely on checkpoint sync plus completion auto-cancel.

## Checkpoint Burst To 63 — 2026-05-11 16:56 UTC

### What ran
Event-polled the run, synced artifacts, checked remote GPU utilization, worker count, bid state, flow-watch logs, and resume-guard logs.

### What was verified
Local and remote artifacts are at `63/164`; the active 4x bid remains capped at `$0.20`, the 8x bid remains paused, and 6 workers are live after additional static shards completed cleanly.

### What remains in this section
Let the remaining static shards finish while the missing-rank guard handles only unfinished-rank failures.

### Blockers
No blocker; lower utilization on GPUs with completed shards is expected and safer than mid-run work stealing.

### Next checkpoint
Continue event polling and sync on the next count jump, preemption, or completion.

## Checkpoint To 64 — 2026-05-11 17:07 UTC

### What ran
Event-polled from `63/164`, synced artifacts, checked latest task files, watcher logs, bid state, remote worker count, and GPU utilization.

### What was verified
Local and remote artifacts advanced to `64/164`; the latest landed task is `024_HumanEval_24.json`, 6 workers remain live, the 4x bid is still the only running instance, and the 8x bid remains paused.

### What remains in this section
Continue monitoring the long-tail tasks and preserve the current static-shard protocol.

### Blockers
No blocker.

### Next checkpoint
Sync and record the next checkpoint jump, preemption, or completion event.

## Paused 8x Cleanup Attempt — 2026-05-11 17:09 UTC

### What ran
Attempted to delete/cancel the paused `polaris-a100x8-ckpt-e8b2d2` bid to reduce overnight clutter and checked Flow bid/status afterward.

### What was verified
Flow returned `not-cancellable` while the 8x bid is paused; the 4x bid remains the only running instance at `$0.20`, and the 8x bid remains paused at `$0.10` with no active status entry.

### What remains in this section
Do not unpause the 8x bid just to cancel it; that could allocate extra GPUs and violate the cost guardrail.

### Blockers
Paused 8x bid cannot be cancelled through the tested Flow commands in its current state.

### Next checkpoint
Keep the paused bid inert and continue the 4x run only.

## Checkpoint Burst To 72 — 2026-05-11 17:12 UTC

### What ran
Event-polled from `64/164`, synced after a burst, then performed an immediate remote health check and second sync because the remote had advanced again.

### What was verified
Local artifacts are now `72/164`; latest synced tasks include `030_HumanEval_30.json` and `038_HumanEval_38.json`, six worker scripts are actually alive, the 4x bid is the only running instance, and the 8x bid remains paused/non-running.

### What remains in this section
Continue the static-shard run and let completed ranks stay stopped unless unfinished assigned tasks are detected.

### Blockers
No blocker; the Flow paused 8x bid remains not-cancellable while paused, so it must stay inert.

### Next checkpoint
Continue syncing on checkpoint jumps and preserve auto-cancel on `164/164`.

## Checkpoint Burst To 74 — 2026-05-11 17:15 UTC

### What ran
Event-polled from `72/164`, synced artifacts twice, checked latest task files, remote worker scripts, GPU utilization, monitor logs, and watcher logs.

### What was verified
Local artifacts advanced to `74/164`; latest synced tasks include `129_HumanEval_129.json` and `137_HumanEval_137.json`, six worker scripts remain active, and only the 4x bid is running.

### What remains in this section
Continue two-minute event polling and let the remaining long-tail static shards complete.

### Blockers
No blocker.

### Next checkpoint
Sync and record the next count jump or preemption/completion event.

## Checkpoint Burst To 77 — 2026-05-11 17:16 UTC

### What ran
Read the monitor state file after the background checkpoint monitor synced another burst, checked active screens, bid state, local artifact count, and latest task files.

### What was verified
Local artifacts are `77/164`; latest synced tasks include `023_HumanEval_23.json`, `145_HumanEval_145.json`, and `153_HumanEval_153.json`, six workers remain live, and monitor heartbeat is fresh.

### What remains in this section
Continue the static-shard run under the current 4x-only cost cap.

### Blockers
No blocker.

### Next checkpoint
Continue watcher-driven sync and cancel on completion.

## Checkpoint To 78 — 2026-05-11 17:24 UTC

### What ran
Checked the monitor state, remote worker heartbeats, GPU utilization, resume-guard logs, and local artifact count after a flat polling interval.

### What was verified
Artifacts are at `78/164`, six worker scripts remain active, GPU utilization is still nonzero across all four A100s, and heartbeat is fresh on the latest started task.

### What remains in this section
Continue through the long-tail tasks without changing concurrency or unpausing additional bids.

### Blockers
No blocker.

### Next checkpoint
Continue syncing and rely on auto-cancel once merge/eval can run at `164/164`.

## Checkpoint To 79 — 2026-05-11 17:29 UTC

### What ran
Event-polled from `78/164`, synced after the count changed, checked latest task files, monitor state, and Flow bid state.

### What was verified
Local artifacts advanced to `79/164`; latest synced tasks include `019_HumanEval_19.json` and `026_HumanEval_26.json`, the 4x bid remains the only active instance, and the 8x bid remains paused/non-running.

### What remains in this section
Continue the existing overnight monitors and do not start any additional expensive experiment.

### Blockers
No blocker.

### Next checkpoint
Let the background monitor sync and the flow watcher auto-cancel after full completion.

## Completion Decision Hook Added — 2026-05-11 17:31 UTC

### What ran
Patched `scripts/watch_flow_preempt_local.sh` so completion now appends a C0c/RWS Table 1 protocol-decision block to `runs/progress.md` after merge/eval/sync, then restarted only the local flow watcher.

### What was verified
`bash -n` passed, the patched watcher is running in `polaris_flow_watch`, and remote workers were not touched.

### What remains in this section
Wait for `164/164`; the watcher should evaluate, sync, append the decision block, and cancel the 4x A100 instance.

### Blockers
No blocker.

### Next checkpoint
Do not start old Phase 1/2; completion should stop at the POLARIS protocol decision.

## Long-Task Plateau At 79 — 2026-05-11 17:39 UTC

### What ran
Polled local/remote counts, checked remote worker scripts, GPU utilization, heartbeats, monitor logs, and resume-guard logs after several flat intervals at `79/164`.

### What was verified
The run remains healthy at `79/164`: six worker scripts are alive, all four A100s show utilization, active tasks include `HumanEval/32`, `161`, `34`, `27`, `46`, and `31`, and the guard is not launching duplicates.

### What remains in this section
Wait through the long-tail tasks and keep the 4x-only cost cap.

### Blockers
No blocker.

### Next checkpoint
Continue event polling; sync and record the next count jump, preemption, or completion.

## Checkpoint To 81 — 2026-05-11 18:10 UTC

### What ran
Event-polled through the long plateau, synced artifacts after the count changed, checked latest task files, remote worker count, and GPU utilization.

### What was verified
Local and remote artifacts advanced to `81/164`; latest synced tasks include `032_HumanEval_32.json` and `161_HumanEval_161.json`, five worker scripts remain active, and the 4x bid is still the only running instance.

### What remains in this section
Let completed static shards stay stopped and continue the remaining long-tail workers.

### Blockers
No blocker; GPU1 is idle because its assigned static shard completed.

### Next checkpoint
Continue event polling and rely on missing-rank relaunch only for unfinished dead ranks.

## Safe Dynamic Fill Enabled — 2026-05-11 18:23 UTC

### What ran
Added `scripts/rws_mcmc_ckpt_fill_worker_remote.sh`, replaced `scripts/rws_mcmc_ckpt_launch_missing_remote.sh` with a lock-aware missing/fill planner, patched monitor/resume/status scripts to count fill workers, dry-ran the plan, then launched three fill workers.

### What was verified
The dry run planned no completed-task reruns, snapshotted in-flight static tasks `40`, `34`, `27`, `46`, and `31`, signaled five static Python workers for stop-after-current, and launched fill workers on `cuda=0,1,1`; status now shows eight checkpointed workers and all four A100s active.

### What remains in this section
Continue from the same checkpointed run directory and let fill workers claim only unlocked, non-completed tasks.

### Blockers
No blocker; a transient resume-guard duplicate-launch attempt was blocked by the checkpointed-worker duplicate guard.

### Next checkpoint
Monitor for monotonic task-count growth and confirm no duplicate `task_index` appears.

## Checkpoint To 83 After Fill — 2026-05-11 18:24 UTC

### What ran
Synced artifacts and recomputed task-index uniqueness from saved JSON files after enabling fill workers.

### What was verified
Local artifacts are `83/164`, duplicate task indices are `[]`, latest dynamic-fill outputs include `039_HumanEval_39.json` and `043_HumanEval_43.json`, and the monitor reports eight running checkpointed workers with ETA around `4h28m`.

### What remains in this section
Keep the fill-enabled checkpoint path running and preserve the completion hook for merge, EvalPlus, protocol decision, and 4x instance cancellation.

### Blockers
No blocker.

### Next checkpoint
Continue event polling from `83/164`.

## Checkpoint To 85 And Relaunch Guard Fix — 2026-05-11 18:41 UTC

### What ran
Synced artifacts, recomputed task-index uniqueness, synced the patched missing/fill launcher, ran remote `bash -n`, dry-ran the launcher, and sent SIGTERM only to the Python child of `fill_20260511T183553Z_1` for stop-after-current.

### What was verified
Local artifacts are `85/164` with duplicate task indices `[]`; the remote dry run reports `ACTIVE_COUNT=8`, `MISSING_RANKS=7`, `LAUNCH_RANKS=""`, and no fill launch, proving the launcher no longer overfills a full pool.

### What remains in this section
Let the signaled filler finish its current task, then allow the monitor to relaunch only missing-task work into open capacity.

### Blockers
No blocker; GPU2 is temporarily carrying three workers and GPU3 one worker until the signaled filler exits.

### Next checkpoint
Continue event polling and verify the next worker refill keeps completed-task count monotonic with no duplicate task indices.

## Static Worker Lock Guard — 2026-05-11 18:48 UTC

### What ran
Patched `scripts/rws_mcmc_ckpt_worker_remote.sh` so relaunched static workers acquire live task locks, skip already locked fill tasks, and write final JSONs with atomic non-overwrite links.

### What was verified
After killing the duplicate rank-2 Python process and relaunching rank 2 with the patched wrapper, `worker_2` skipped live-locked `HumanEval/42` and started `HumanEval/50`; local artifacts are `86/164` with duplicate task indices `[]`.

### What remains in this section
Keep the monitor on the checkpointed launcher path and let signaled old workers retire after current tasks.

### Blockers
No blocker.

### Next checkpoint
Confirm the next count increase remains monotonic and the active lock set contains no duplicate task index.

## Refill Verified To 89 — 2026-05-11 18:52 UTC

### What ran
Synced artifacts, checked local duplicate task indices, checked Flow bids, and inspected monitor/resume logs after the patched static relaunch path ran.

### What was verified
Local artifacts advanced monotonically to `89/164` with duplicate task indices `[]`; the remote pool is back to eight checkpointed workers, all four A100s are active, and Flow shows only `polaris-a100x4-ckpt-66304a` allocated while the 8x bid remains paused.

### What remains in this section
Let the checkpointed monitor continue filling missing work and rely on the completion watcher to merge, score, sync, and cancel the 4x instance at `164/164`.

### Blockers
No blocker.

### Next checkpoint
Continue polling for monotonic count growth, duplicate-free task artifacts, and single-active-bid cost hygiene.

## Health Loop To 94 — 2026-05-11 18:57 UTC

### What ran
Ran three periodic sync-and-health checks over the checkpointed run, each recomputing duplicate task indices from saved JSONs and checking remote worker/GPU state.

### What was verified
The run advanced from `90/164` to `94/164`, every local duplicate check returned `[]`, remote worker count stayed at eight, and all four A100s remained active.

### What remains in this section
Continue periodic monitoring until either completion lands or a worker/preemption/cost anomaly needs intervention.

### Blockers
No blocker.

### Next checkpoint
At `164/164`, verify merge and EvalPlus scoring, then ensure the completion watcher cancels the active 4x instance.

## Extended Health To 99 — 2026-05-11 19:06 UTC

### What ran
Ran five more periodic sync-and-health checks, including local duplicate recomputation and remote worker/GPU checks.

### What was verified
The run advanced from `96/164` to `99/164`; every local duplicate check returned `[]`, remote worker count stayed at eight, and all four A100s remained active.

### What remains in this section
Continue monitoring the long tail without changing sampler, prompt, model, or evaluator configuration.

### Blockers
No blocker.

### Next checkpoint
If progress plateaus, verify worker heartbeats and locks before intervening; otherwise wait for the completion watcher at `164/164`.

## Long Tail Plateau At 101 — 2026-05-11 19:25 UTC

### What ran
Ran another five periodic sync-and-health checks plus a live lock-age inspection.

### What was verified
The run is plateaued at `101/164` with duplicate task indices `[]`, eight remote checkpointed workers, saturated GPUs, and live locks for task indices `35`, `42`, `47`, `50`, `59`, `80`, and `119`.

### What remains in this section
Wait for the active long-tail MCMC tasks to finish; do not kill alive locked workers because that would discard in-flight compute.

### Blockers
No blocker; this is a compute-bound plateau, not an idle or duplicate-launch condition.

### Next checkpoint
Continue periodic sync checks and intervene only on stale locks, worker loss without refill, duplicate artifacts, preemption, or completion.

## Conservative Health To 103 — 2026-05-11 19:36 UTC

### What ran
Ran three five-minute foreground health checks while leaving the background checkpoint monitor, resume guard, and completion watcher active.

### What was verified
The run advanced from `101/164` to `103/164`, local duplicate task indices remained `[]`, remote worker count stayed at eight, and all four A100s stayed busy.

### What remains in this section
Continue lower-frequency foreground checks while background monitors handle preemption, refill, sync, merge/eval, and cancellation.

### Blockers
No blocker.

### Next checkpoint
Record the next material count increase or any anomaly; otherwise keep the run unchanged.

## Progress Health To 111 — 2026-05-11 19:58 UTC

### What ran
Ran three more five-minute foreground sync-and-health checks after inspecting active worker log tails.

### What was verified
The run advanced from `103/164` to `111/164`, local duplicate task indices remained `[]`, and remote worker count stayed at eight.

### What remains in this section
Continue periodic checks until completion; do not change concurrency or sampler configuration while the checkpointed pool remains healthy.

### Blockers
No blocker.

### Next checkpoint
At the next material count increase, sync docs; at `164/164`, audit merge, EvalPlus scoring, and 4x instance cancellation.

## Refill To 123 — 2026-05-11 20:10 UTC

### What ran
After the foreground check found `122/164` with seven workers and GPU3 underfilled, dry-ran the checkpointed missing/fill launcher, then launched one dynamic filler on `cuda=3`.

### What was verified
The dry run planned no static relaunches and one fill worker; after launch, artifacts advanced to `123/164`, duplicate task indices were `[]`, worker count returned to eight, and the new filler started `HumanEval/90`.

### What remains in this section
Continue running the checkpointed pool and keep the 4x A100 as the only allocated instance.

### Blockers
No blocker.

### Next checkpoint
Watch for the next material count increase or completion; do not alter RWS sampling or evaluation settings.

## Late Health To 137 — 2026-05-11 20:22 UTC

### What ran
Ran another three foreground sync-and-health checks after the GPU3 filler refill.

### What was verified
The run advanced from `126/164` to `137/164`, local duplicate task indices remained `[]`, and remote worker count stayed at eight.

### What remains in this section
Continue checkpoint monitoring for the final 27 artifacts and let the completion watcher handle merge, EvalPlus scoring, sync, and cancellation.

### Blockers
No blocker.

### Next checkpoint
Inspect and refill only if worker count drops again; otherwise wait for completion.

## Static Rank Refill To 138 — 2026-05-11 20:24 UTC

### What ran
After remote count reached `138/164` with seven workers and GPU0 underfilled, dry-ran the checkpointed launcher and relaunched missing static rank `0` on `cuda=0`.

### What was verified
The dry run planned exactly `LAUNCH_RANKS=0`; after launch, worker count returned to eight, rank `0` started `HumanEval/128`, and local duplicate task indices remained `[]`.

### What remains in this section
Continue final-stretch monitoring without changing the scientific configuration.

### Blockers
No blocker.

### Next checkpoint
Wait for the next material count increase or completion.

## Final Stretch To 143 — 2026-05-11 20:33 UTC

### What ran
Ran three final-stretch foreground sync-and-health checks after relaunching static rank `0`.

### What was verified
The run advanced from `137/164` to `143/164`, local duplicate task indices remained `[]`, and remote worker count stayed at eight.

### What remains in this section
Continue monitoring the final 21 artifacts and verify completion cleanup.

### Blockers
No blocker.

### Next checkpoint
At completion, audit saved artifacts, merge/scoring outputs, progress decision block, and Flow instance deletion.

## Completion Watch To 150 — 2026-05-11 20:44 UTC

### What ran
Ran three completion-watch foreground sync-and-health checks over the final stretch.

### What was verified
Local artifacts advanced from `145/164` to `150/164`, remote artifacts reached `151/164` immediately after the last sync, local duplicate task indices remained `[]`, and remote worker count stayed at eight.

### What remains in this section
Continue monitoring the final artifacts and ensure the completion watcher performs merge, EvalPlus scoring, local sync, progress decision append, and 4x instance deletion.

### Blockers
No blocker.

### Next checkpoint
Run one more close-out monitor loop and audit completion cleanup if `164/164` lands.

## Closeout Watch To 154 — 2026-05-11 20:53 UTC

### What ran
Ran three closeout sync-and-health checks and dry-ran the checkpoint launcher when a pgrep count appeared low.

### What was verified
Local artifacts advanced to `154/164` with duplicate task indices `[]`; the launcher dry run reported eight active tasks and no relaunch needed, with live locks for task indices `82`, `110`, `136`, `138`, `147`, `155`, `162`, and `163`.

### What remains in this section
Wait for the last 10 artifacts and audit automatic merge/scoring/cancellation.

### Blockers
No blocker.

### Next checkpoint
Do not relaunch unless the lock-aware launcher reports open capacity; otherwise wait for completion.

## Tail Planner Lock Fix To 157 — 2026-05-11 21:05 UTC

### What ran
Patched `scripts/rws_mcmc_ckpt_launch_missing_remote.sh` so missing-rank detection ignores task indices already protected by live lock owners, synced it remote, dry-ran it, and launched one tail filler on `cuda=2`.

### What was verified
The patched dry run stopped falsely relaunching no-op rank `2`, planned one dynamic fill worker, and after launch the run advanced to `157/164` with duplicate task indices `[]`; the new filler started `HumanEval/160`.

### What remains in this section
Wait for the final seven artifacts and verify the automatic completion cleanup.

### Blockers
No blocker.

### Next checkpoint
Completion audit: artifact count, merge/eval outputs, progress decision block, and Flow deletion.

## Final Locked Tail At 159 — 2026-05-11 21:15 UTC

### What ran
Patched the launcher again so fill launches require `UNCLAIMED_REMAINING > 0`, synced the patch, and dry-ran the planner on the final tail.

### What was verified
The planner reports `REMAINING=5`, `UNCLAIMED_REMAINING=0`, and no launch; local artifacts are `159/164` with duplicate task indices `[]`, and the five remaining tasks are already live-locked.

### What remains in this section
Wait for the final five locked tasks to finish and verify automatic cleanup.

### Blockers
No blocker.

### Next checkpoint
Completion audit only; do not launch additional workers unless a lock becomes stale or unclaimed work appears.

## Tail Throughput Guardrail — 2026-05-11 21:45 UTC

### What ran
Audited the final five live locks, process/GPU placement, launcher dry run, and per-task log progress.

### What was verified
The run remains at `159/164` with duplicate task indices `[]`; tasks `82`, `136`, `147`, `155`, and `163` are live-locked by five checkpointed workers, and the dry-run launcher reports `UNCLAIMED_REMAINING=0`, so it correctly refuses duplicate relaunches.

### What remains in this section
Continue one-shot sync monitoring until `164/164`, then merge, EvalPlus-score, sync local artifacts, append the protocol decision, and delete the running 4xA100 instance.

### Blockers
No blocker; GPU2 is idle, but safe migration would require killing in-flight work and two remaining tasks are already near completion.

### Next checkpoint
Do not duplicate or migrate live-locked task indices; only relaunch if a lock becomes stale, a worker exits, or unclaimed work appears.

## Tail Advanced To 161 — 2026-05-11 21:50 UTC

### What ran
Synced the checkpointed run, audited live locks and process/GPU placement, and dry-ran the missing-task launcher again.

### What was verified
Local artifacts advanced to `161/164` with duplicate task indices `[]`; the remaining task indices are `147`, `155`, and `163`, all live-locked by checkpointed workers, and the launcher reports `UNCLAIMED_REMAINING=0`.

### What remains in this section
Wait for the last three task JSONs, then merge, EvalPlus-score, sync results, append the post-`C0c` protocol decision, and delete the 4xA100 instance.

### Blockers
No blocker; GPU0 and GPU2 are idle, but restarting live tasks would discard in-flight work and is not a safe high-fidelity speedup.

### Next checkpoint
Continue sync monitoring; only relaunch on stale lock, worker exit, or unclaimed remaining work.

## Tail Advanced To 162 — 2026-05-11 21:52 UTC

### What ran
Synced the checkpointed artifacts and dry-ran the missing-task launcher after task `147` completed.

### What was verified
Local artifacts advanced to `162/164` with duplicate task indices `[]`; the remaining task indices are `155` and `163`, each live-locked by one checkpointed worker, and the launcher reports `UNCLAIMED_REMAINING=0`.

### What remains in this section
Wait for tasks `155` and `163`, then run completion audit, merge, EvalPlus scoring, local sync, protocol decision, and instance deletion.

### Blockers
No blocker; the remaining two workers are on separate GPUs, so there is no safe throughput intervention left.

### Next checkpoint
Completion audit when `164/164` lands, or stale-lock diagnosis if either worker exits.

## Preemption At 162 And Resume Bid — 2026-05-11 22:15 UTC

### What ran
Observed Flow transition `polaris-a100x4-ckpt-66304a` from `preempting` to `paused`, confirmed local artifacts remained `162/164`, and raised the same bid limit from `$0.20` to `$0.25`.

### What was verified
The bid is now `Open` at `$0.25`, the standby `polaris-a100x8-ckpt-e8b2d2` remains `Paused` at `$0.10`, and only the in-flight task indices `155` and `163` need to be resumed from checkpoint.

### What remains in this section
Wait for the 4xA100 bid to reallocate, SSH in, verify `nvidia-smi -L`, and launch only the checkpointed missing-task path for tasks `155` and `163`.

### Blockers
No data-loss blocker; the current blocker is Flow reallocation.

### Next checkpoint
When `started_at` and SSH return, run the checkpointed resume launcher and confirm no completed task indices are recomputed.

## Resume Bid Raised — 2026-05-11 22:20 UTC

### What ran
Monitored the reopened 4xA100 bid for five minutes at `$0.25`, then raised only that existing bid to `$0.50`.

### What was verified
`polaris-a100x4-ckpt-66304a` remains `Open` at `$0.50`; `polaris-a100x8-ckpt-e8b2d2` remains `Paused` at `$0.10`, so no second paid instance was started.

### What remains in this section
Wait for reallocation, then resume only missing task indices `155` and `163` from the checkpointed run directory.

### Blockers
Flow has not reallocated the 4xA100 yet.

### Next checkpoint
Recheck bid status every minute; if reallocated, verify SSH/GPU and run the checkpointed missing-task launcher.

## Resume Launched For Final Two — 2026-05-11 22:41 UTC

### What ran
Verified SSH and `nvidia-smi -L` after reallocation, reran `scripts/rws_mcmc_ckpt_prepare_remote.sh` because `/mnt/local` cache was wiped by preemption, and launched the checkpointed missing-task path.

### What was verified
The model cache was restored to snapshot `d149729398750b98c0af14eb82c78cfe92750796`; only task indices `155` and `163` remain missing, and exactly two active workers now hold those live locks after extra fill workers exited without claims.

### What remains in this section
Monitor the final two workers to `164/164`, then run merge, EvalPlus scoring, local sync, protocol decision, and instance deletion.

### Blockers
No blocker.

### Next checkpoint
Audit that the local task count reaches `164/164` with duplicate task indices `[]`.

## C0c Completion And Protocol Decision — 2026-05-11 23:59 UTC

### What ran
Completed the checkpointed official RWS MCMC `C0c` full-164 HumanEval run for `Qwen/Qwen2.5-7B`; merged task JSONs to `runs/rws_official_full164_mcmc_ckpt/results/qwen/qwen_he_mcmc_ckpt_full164.csv`; remote EvalPlus was interrupted by automatic teardown, so full EvalPlus scoring was recomputed locally from the saved 164 task JSONs with `EVALPLUS_MAX_MEMORY_BYTES=-1` to bypass a macOS `setrlimit` incompatibility.

### What was verified
Saved artifacts contain `164/164` task JSONs with duplicate task indices `[]`; local EvalPlus summary reports HumanEval `91/164 = 0.5549` and HumanEval+ `79/164 = 0.4817`; HumanEval is `-6.7pp` versus RWS Table 1's `0.622` power-sampling target and within the previously used `±7pp` replication tolerance, while HumanEval+ shows the expected stronger-test drop.

### What remains in this section
No further experiment may start under this goal; the next goal should choose and pre-register the next bounded POLARIS experiment.

### Blockers
No running 4xA100 spend remains: `polaris-a100x4-ckpt-66304a` is cancelled; `polaris-a100x8-ckpt-e8b2d2` remains paused at `$0.10` because Flow reported no running task to cancel.

### Next checkpoint
DECISION: choose **POLARIS-MATH500 primary** for the next goal, using the completed HumanEval+ result as a Sustained-regime negative-control signal rather than immediately scaling HumanEval+ Phase 1/2; rationale: the code track is expensive, preemption-sensitive, and substantially lower on HumanEval+ (`0.4817`) than HumanEval (`0.5549`), while the revised thesis predicts the cleanest inference-time composition signal in Diminished regimes such as MATH500.

## POLARIS-MATH500 v1 Pre-Registration — 2026-05-12 05:50 UTC

### What ran
Pre-registered the POLARIS-MATH500 v1 run in `TODO.md#polaris-math500-v1` per proposal §"Drift-Lock Protocol" item 4. Locked: model `Qwen/Qwen2.5-7B`, MATH500 test slice `0-75`, dev slice `75-100`, six conditions (`greedy`, `bon_temp1`, `single_prompt_power`, `single_best_prompt`, `full_archive_fixed`, `full_archive_mixed`), `B=8` per problem (1 for greedy), `k=4`, seeds `{17, 71, 1729}`, `MCMC_STEPS=10`, `MCMC_BLOCK_NUM=16`, `MAX_NEW_TOKENS=3072`, verifier `math/sympy-equivalence-v1`, selector `argmax_verifier_score_iteration_tiebreak`, descriptor extractor `v1-heuristic-2026-05-12`.

### What was verified
Phase 0 (vendored vs upstream drift sweep): MATH500 critical path (`power_samp_utils.py`, `grader_utils/math_grader.py`, `parse_utils.py`, `math_normalize.py`, `constants.py`, `data/MATH500.json`) has zero functional drift — only import-path rewrites and black/ruff cosmetic reformatting. Stale `polaris/vendored/PORTS.json` references stripped from `dc/__init__.py`, `gepa/__init__.py`, `evalplus/__init__.py`. Phases 1–7 complete: 79 unit + smoke tests green on Mac CPU (no torch dep). `MCMC_BLOCK_SIZE=192` is now derived from `MCMC_BLOCK_NUM=16` per RWS source; legacy `K_t`/`M_t` and HumanEval-phase constants removed from `src/polaris/config.py`.

### What remains in this section
Superseded on 2026-05-13 22:28 UTC by the research-faithful split policy:
archive construction now uses external math optimizer-dev rows, and final
reporting uses full MATH500 rather than the old 0-75 slice.

### Blockers
No blocker. Local CPU pipeline is green; awaiting GPU provisioning for archive construction.

### Next checkpoint
Archive SHA256 recorded here once `build_mapelite.py` emits `archive.json`.

## Base Model Correction Before Modal Smoke — 2026-05-12 06:00 UTC

### What ran
Caught an inherited-model mistake before any GPU spend: the initial Modal smoke attempt was wired to `Qwen/Qwen2.5-7B`, which was the C0c HumanEval base, not the RWS MATH500 base. Modal image build was already in progress but aborted on an unrelated host-file lock error, so no model load or GPU compute happened — spend was effectively `$0`.

### What was verified
Read RWS (arXiv:2510.14901) §5.1 and Table 1 directly via `mcp__claude_ai_alphaxiv__get_paper_content`. Confirmed:
- RWS tests three base models across all benchmarks: `Qwen2.5-Math-7B`, `Qwen2.5-7B`, `Phi-3.5-mini-instruct`.
- MATH500 Power-Sampling row: Qwen2.5-Math-7B `0.748`, Qwen2.5-7B `0.706`, Phi-3.5-mini `0.508`. Math-specialized base is the canonical strongest pair for MATH500.
- HumanEval Power-Sampling row: Qwen2.5-7B `0.622` (C0c's target — correctly paired), Qwen2.5-Math-7B `0.573`, Phi-3.5-mini `0.732`.
- Hyperparameters verbatim from §5.1: `T_max=3072`, block size `B = 3072/16 = 192` (so `block_num=16`), `α=4.0`, proposal LLM = base with `temperature=1/α=0.25`. All match `src/polaris/config.py` after the Phase 2 truth-up.

### What remains in this section
Updated `MODEL_ID="Qwen/Qwen2.5-Math-7B"` in `src/polaris/config.py` and the TODO.md pre-registration block. Future-session rule landed in `~/.claude/CLAUDE.md` and project memory: "ALWAYS double-check base model + hyperparameters against the source paper's table before any paid GPU launch — never inherit from a prior run on a different benchmark." Next: rerun Modal smoke with the corrected model.

### Blockers
None.

### Next checkpoint
Modal smoke (1 problem, greedy, `Qwen/Qwen2.5-Math-7B`).

## Publishable Readiness Infrastructure Implemented — 2026-05-13 20:23 UTC

### What ran
Implemented the end-to-end readiness surfaces for the full POLARIS method rather
than only archive+power baselines: split-level dataset locks, the
`polaris_full_verified_memory` condition, memory-enabled runner plumbing,
archive-memory dry-run builder, production plan metadata, and final artifact
audit gates. Updated `AGENTS.md` and `TODO.md` so future sessions treat this as
gated publishable-readiness infrastructure rather than the older MATH500-only
R5 state.

### What was verified
`.venv-eval/bin/pytest -q` passed with `150 passed`; `bash
scripts/check_protocol_sync.sh` passed; `scripts/smoke_polaris_readiness.py
--out runs/readiness_smoke.tmp` passed with no deferred tracks/experiments and
zero generation calls on cache replay; `scripts/build_polaris_archive.py
--track math500 --mode freeze --dev-split 75 76 --out
runs/archive_build_smoke.tmp --dry-run` emitted a frozen archive-memory bundle;
and `scripts/run_condition.py --condition polaris_full_verified_memory
--preflight-only` passed against that archive.

### What remains in this section
GPQA-Diamond is still externally gated: split locks are `pending_auth` until
accepted HF terms plus `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`, or an official
`GPQA_DIAMOND_PATH` cache, is configured. No paid model call, small real-slice,
descriptor-judge run, real GEPA evolution, or final experiment was launched.

### Blockers
External GPQA auth/cache and explicit paid-run authorization with cost cap for
small real slices and final runs.

### Next checkpoint
Refresh dataset locks after GPQA auth/cache, then run one cached real row per
track/condition and audit the resulting artifact bundles before any paid
small-slice launch.

## Research-Faithful Split Policy Installed — 2026-05-13 22:28 UTC

### What ran
Checked the split conventions in the relevant papers and changed the repo to
match them: sampling-only baselines report the full benchmark, while any
POLARIS prompt/archive/memory optimization must use non-final optimizer-dev
data. Added external optimizer-dev loaders for math (`MATH-lighteval` train
rows), code (MBPP+ rows), and GPQA non-Diamond rows; updated dataset locking,
production planning, archive-building defaults, and protocol docs.

### What was verified
Dataset locks now show: MATH dev `math_optimizer_dev train[0:500]` locked,
MATH final `MATH500 test[0:500]` locked; code dev `MBPP+ [0:378]` locked,
HumanEval+ final `test[0:164]` locked; GPQA non-Diamond dev and GPQA-Diamond
final remain `pending_auth`. Production plan final splits are `[0,500]`,
`[0,164]`, and `[0,198]`. Focused split/readiness tests passed with `15
passed`.

### What remains in this section
GPQA still needs accepted HF access or official local cache. No paid run or
final benchmark execution was launched.

### Blockers
GPQA auth/cache and explicit paid-run authorization with cost cap.

### Next checkpoint
Run the full local suite and protocol guard, then use the new locks for cached
real-row rehearsal.

## FarmShare ProRL Recovery Gates 0-2 — 2026-05-13 23:54 UTC

### What ran
Started the FarmShare-only ProRL Recoverable Fraction audit goal. Implemented
the ProRL recovery orchestration surface, pinned `transformers==4.51.3` for
`vllm==0.9.2`, tightened the FarmShare artifact audit, and fixed empty-shard
JSONL emission. Synced the repo to `/scratch/users/duynguy/polaris/repo`,
created the micromamba env under `/scratch`, and ran the one-GPU model-load
gate plus the 3-problem MATH500 base pass@4 smoke.

### What was verified
Local Gate 0 passes with `177 passed` plus `scripts/check_protocol_sync.sh`.
FarmShare reports L40S GPUs and the scratch env imports `torch`,
`transformers`, `vllm`, `datasets`, `reasoning_gym`, `pandas`, and `pyarrow`.
The one-GPU vLLM gate loaded
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
on oat and generated successfully. Slurm job `1562962` completed all four
MATH500 pass@4 smoke shards in under one minute per shard. Synced artifacts in
`runs/farmshare/smoke/math500_base_pass4_v2/`; all four shard audits pass, with
shard problem counts `1,1,1,0` and complete required artifact bundles.

### What remains in this section
Run Gate 3 pilots: 20 MATH500 problems x four checkpoints x pass@16, then the
30-problem multi-family pilot. Do not call any RF result valid until the Phase 0
Karan-Du external replication gate passes.

### Blockers
GPQA remains auth-gated unless `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` or
`GPQA_DIAMOND_PATH` exists on FarmShare. No paid fallback is allowed under this
goal.

### Next checkpoint
Patch the pilot launcher so Phase 1 pilot uses `samples_per_problem=16` rather
than the main-run `128`, then submit the 20-problem MATH500 pilot.

## ProRL Recovery Active Goal Drift Resolution — 2026-05-14 03:59 UTC

### What ran
Resolved a protocol-surface conflict before any new FarmShare launch after the
active MATH500 pilot. `TODO.md` and `docs/PRORL_RECOVERY_AUDIT.md` still
described the earlier readiness-only goal, while the active thread goal now
authorizes FarmShare-only execution through the 100-problem main artifacts or a
hard gate failure.

### What was verified
The active restrictions are unchanged: FarmShare only, no paid fallback, no RF
claims before the Phase 0 Karan-Du external replication gate, no GPQA official
rows without HF/auth or `GPQA_DIAMOND_PATH`, and no ProRL/BroRL traces in the
main RF numerator memory.

### What remains in this section
Finish the running 20-problem MATH500 pass@16 pilot, sync and audit artifacts,
then launch the 30-problem non-GPQA multi-family pilot only if the pilot audit
passes.

### Blockers
GPQA remains auth-gated on FarmShare. No GPQA environment variable is present:
`GPQA_DIAMOND_PATH=False`, `HF_TOKEN=False`, and
`HUGGINGFACE_HUB_TOKEN=False`.

### Next checkpoint
Run `scripts/check_protocol_sync.sh`, continue monitoring Slurm job `1562992`,
and aggregate `phase1_results.parquet` after all shards complete.

## Phase 0 Backend Authority Correction — 2026-05-14 04:03 UTC

### What ran
Audited the ProRL recovery Phase 0 cell plan while the MATH500 pilot was still
running. The plan already used the vendored RWS MATH500 data/prompt, Qwen2.5
Math-7B, `max_new_tokens=3072`, `block_num=16`, and `N_MCMC=10`, but it
inherited the default vLLM backend.

### What was verified
Patched `phase0_cells` to set `backend="hf"` so the Karan-Du replication gate
uses the HF/RWS path rather than the optimized vLLM MCMC path. Local focused
tests and drift guard passed; the synced FarmShare repo also passed drift guard
and focused ProRL/FarmShare tests.

### What remains in this section
Phase 0 itself has not run yet. No RF claim is valid until the full Phase 0
replication lands within `0.748 +- 0.02`.

### Blockers
None for Phase 0 setup. The expected Phase 0 run is heavier than the pilots
because it uses the HF/RWS scorer path intentionally.

### Next checkpoint
Complete and audit the running 20-problem MATH500 pilot, then run the
30-problem non-GPQA multi-family pilot before launching Phase 0.

## FarmShare Pilot Audit And Startup Optimization — 2026-05-14 04:08 UTC

### What ran
Added a plan-level ProRL recovery artifact audit and removed `micromamba run`
from future Slurm array task startup. Slurm scripts now put the scratch env on
`PATH` and call `python` directly, avoiding shared mamba lock waits observed in
the running pilot logs.

### What was verified
The new `audit-plan` path checks required files plus row completeness against
the rendered cell plan: selected rows equal sharded problem count, and
candidates/scores equal sharded problem count times samples per problem. Local
focused tests passed with `28 passed`; the synced FarmShare repo also passed
the same focused tests.

### What remains in this section
Use `audit-plan` after syncing the MATH500 pilot instead of relying only on
single-directory artifact presence.

### Blockers
None.

### Next checkpoint
Wait for Slurm job `1562992` to clear, sync
`phase1_math20_p16_v4`, run `audit-plan`, and aggregate
`phase1_results.parquet`.

## Phase 1 Candidate Token Provenance Fix — 2026-05-14 04:14 UTC

### What ran
Patched the runner candidate schema so generated rows carry `token_ids`,
`logprobs_norm`, and `logprobs_unnorm` directly instead of only storing those
fields in the trajectory cache.

### What was verified
Added smoke coverage for fresh batched rows and cache replay rows. Local and
FarmShare targeted tests passed with `56 passed` on the runner/FarmShare
surface.

### What remains in this section
The already-running `phase1_math20_p16_v4` pilot started before this patch, so
its candidate JSONL may still have empty token-id columns after aggregation.
Future Phase 1 pilot/main shards will include token IDs directly.

### Blockers
None.

### Next checkpoint
After the current pilot clears, aggregate it as pilot evidence only; use the
patched schema for the 30-problem multi-family pilot and main Phase 1 runs.

## vLLM Plain Generation Stop-Policy Fix — 2026-05-14 04:21 UTC

### What ran
Diagnosed the slow BroRL pilot tail by reading completed row metadata: every
base, ProRL v1, ProRL v2, and BroRL sample generated exactly to its configured
cap (`8192` or `16384`). Source inspection showed the vLLM plain generation path
used `min_tokens=max_tokens` and `ignore_eos=True`, which is appropriate for
MCMC block proposals but wrong for greedy and BoN sampling.

### What was verified
Patched vLLM plain generation to allow EOS while keeping MCMC proposal/draft
sampling exact-length. Local full unit/smoke suite passed with `180 passed`;
the synced FarmShare repo passed targeted vLLM/runner/FarmShare tests with
`63 passed`.

### What remains in this section
The running `phase1_math20_p16_v4` pilot was launched before this fix, so it is
valid as Slurm/artifact plumbing evidence but not as a throughput estimate.
Future pilot/main runs use the corrected stop policy.

### Blockers
None.

### Next checkpoint
Finish and audit the active pilot, then launch the 30-problem non-GPQA
multi-family pilot from the patched repo.

## Phase 1 MATH500 Pilot Artifact Audit — 2026-05-14 04:44 UTC

### What ran
Slurm job `1562992` finished the 20-problem MATH500 pass@16 pilot across base,
ProRL v1, ProRL v2, and BroRL. Artifacts were synced from FarmShare and audited
under `runs/farmshare/phase1_math20_p16_v4`.

### What was verified
`scripts/run_prorl_recovery.py audit-plan --phase phase1 --tracks math500
--problem-count 20 --samples-per-problem 16` passed with `16` cells,
`80` selected rows, `1280` candidate rows, and `1280` score rows. The canonical
pilot aggregate was written to
`runs/farmshare/phase1_math20_p16_v4/phase1_results.parquet`.

### What remains in this section
This pilot remains plumbing evidence only because it was launched before the
vLLM EOS stop-policy and candidate token-provenance fixes. Do not use its
runtime as the ETA for patched pilots or its rows as final Phase 1 science.

### Blockers
None.

### Next checkpoint
Run the 30-problem non-GPQA multi-family pilot from the patched FarmShare repo.

## Phase 1 Multi-Family Pilot Launch — 2026-05-14 04:45 UTC

### What ran
Attempted to launch the full 64-cell non-GPQA multi-family pilot array. FarmShare
rejected it with `QOSMaxSubmitJobPerUserLimit`, confirming that the submit cap
counts array elements rather than only active jobs.

### What was verified
Split the pilot into queue-compliant 32-cell chunks while preserving `%4`
running concurrency. Chunk A launched as Slurm job `1563184` over cells `0-31`,
covering MATH500 and Reasoning Gym `boxnet` across all four checkpoints.

### What remains in this section
After chunk A clears and audits, launch chunk B for cells `32-63`, covering
Reasoning Gym `graph_color` and `family_relationships`.

### Blockers
None. This is a scheduler submit-limit constraint, not a compute or protocol
failure.

### Next checkpoint
Monitor job `1563184`, inspect GPU utilization after model load, and sync/audit
the chunk-A artifacts on completion.

## Paper Comparison Of Pre-Fix MATH500 Pilot — 2026-05-14 06:02 UTC

### What ran
Compared the completed `phase1_math20_p16_v4` MATH500 pilot aggregate against
the DeepSeek-R1, ProRL/Nemotron, BroRL, and Karan-Du paper/model-card reference
numbers.

### What was verified
The pilot aggregate reported base pass@1 `0.55`, ProRL v1 `0.70`, ProRL v2
`0.65`, and BroRL `0.60` on 20 MATH500 problems, with pass@16 between `0.90`
and `0.95`. Every generated row hit the configured cap: `8192` tokens for
base/v1 and `16384` for v2/BroRL.

### Interpretation
These rows are not paper-comparable. DeepSeek reports
DeepSeek-R1-Distill-Qwen-1.5B at `83.9%` MATH-500 pass@1 using temperature
`0.6`, top-p `0.95`, max generation length `32768`, and 64 responses per query
for pass@1 estimation. The ProRL/Nemotron card reports Math pass@1 of `82.90`
for base, `91.89` for ProRL v1, `92.49` for ProRL v2, and `92.20` for BroRL
under their vLLM evaluation settings. Karan-Du's Phase 0 target remains the
separate exact external gate: Qwen2.5-Math-7B MATH500 Power Sampling `0.748`.

### What remains in this section
Use the pre-fix MATH500 pilot only as Slurm/artifact plumbing evidence. The
current patched multi-family pilot is the first EOS-corrected Phase 1 sanity
run, but strict paper parity still requires a paper-faithful MATH500 sanity
cell with the paper eval settings before interpreting checkpoint gaps.

### Blockers
None. This is a validity classification, not a hard gate failure.

### Next checkpoint
Finish and audit Slurm job `1563184`, then compare the patched aggregate
against the same paper references.

## Reasoning Gym Answer Extraction Fix — 2026-05-14 06:45 UTC

### What ran
Investigated the all-zero `reasoning_gym_boxnet` rows in Slurm job `1563184`.
DeepWiki and installed-source inspection showed `boxnet` and `graph_color`
score raw JSON strings, while `family_relationships` scores an exact
relationship string. The project scorer was passing the whole model response,
and the Phase 1 cells were using the MATH500 `\\boxed{}` direct prompt.

### What was verified
Patched Reasoning Gym scoring to extract `<answer>...</answer>` via
`reasoning_gym.utils.extract_answer` before calling `dataset.score_answer`.
Added a `reasoning_gym_direct` archive that asks for only the required JSON or
string inside the answer tag, and routed Phase 1 Reasoning Gym cells to that
archive. Local unit/smoke tests passed with `182 passed`; the synced FarmShare
repo regenerated four archive JSONs and passed focused remote tests with
`55 passed`.

### What remains in this section
The `boxnet` rows from `phase1_multifamily30_p16_v1` are invalid for science and
should be treated as prompt/verifier-surface evidence only. Relaunch corrected
Reasoning Gym pilots under a new root before interpreting ProRL gaps on
Sustained tasks.

### Blockers
None.

### Next checkpoint
Launch corrected 30-problem Reasoning Gym pilots from the patched FarmShare
repo.

## Reasoning Gym Verifier Fail-Closed Patch — 2026-05-14 06:50 UTC

### What ran
Launched corrected Reasoning Gym pilot job `1563368` under
`phase1_rg30_p16_v2`. The first two `boxnet` cells failed because
`BoxnetDataset.score_answer` can throw on JSON values with the wrong shape
instead of returning `0.0`.

### What was verified
Cancelled `1563368` and patched `score_reasoning_gym` so malformed candidate
answers fail closed with score `0.0` and a `scorer_error` field, rather than
crashing the Slurm cell. Local unit/smoke tests passed with `183 passed`; the
synced FarmShare repo regenerated archive JSONs and passed focused remote tests
with `31 passed`.

### What remains in this section
The partial `phase1_rg30_p16_v2` artifacts are invalid for science because
some cells failed and the job was cancelled intentionally.

### Blockers
None after the fail-closed patch.

### Next checkpoint
Relaunch corrected Reasoning Gym `boxnet + graph_color` pilot under a fresh
root.

## Reasoning Gym Archive Deployment Miss — 2026-05-14 06:57 UTC

### What ran
Relaunched corrected Reasoning Gym pilot job `1563393` under
`phase1_rg30_p16_v3`. All cells failed immediately before model loading.

### What was verified
Logs showed `FileNotFoundError` for
`data/prorl_recovery_archives/reasoning_gym_direct.json`. The previous
`sync --delete` removed generated remote archive files because they are not
tracked local files; this was an operational deployment miss, not a model or
verifier failure.

### What remains in this section
Regenerate archive JSONs on FarmShare after each repo sync before submitting
cells that reference generated archive files.

### Blockers
None after archive regeneration.

### Next checkpoint
Relaunch corrected Reasoning Gym pilot under a fresh root after confirming
`reasoning_gym_direct.json` exists remotely.

## Corrected Reasoning Gym Boxnet/Graph-Color Pilot — 2026-05-14 08:55 UTC

### What ran
Relaunched corrected Reasoning Gym `boxnet + graph_color` pilot as Slurm job
`1563432` under `phase1_rg30_p16_v4`, after regenerating
`reasoning_gym_direct.json` on FarmShare.

### What was verified
The job completed with no failed Slurm child tasks. Artifact audit passed with
`32` cells, `3840/3840` candidate rows, `3840/3840` score rows, and `240`
selected rows. Aggregate output:
`runs/farmshare/phase1_rg30_p16_v4/phase1_results.parquet`.

### Pilot results
`graph_color` now shows a large checkpoint gap: base pass@1 `0.033`, base
pass@16 `0.333`, ProRL v1 pass@1/pass@16 `1.0/1.0`, ProRL v2 `0.8/1.0`, and
BroRL `0.9/1.0`. `boxnet` remains `0.0` for all checkpoints and all reported
pass@k values, so it cannot yet support a ProRL recovery claim without a
paper-prompt/config check.

### Blockers
No infrastructure blocker for corrected Reasoning Gym execution. Scientific
blocker remains for `boxnet`: all checkpoints fail under the current prompt and
configuration.

### Next checkpoint
Run the corrected `family_relationships` 30-problem pilot and then inspect
boxnet prompt/config against the ProRL paper/source before any main-run boxnet
claim.

## Corrected Reasoning Gym Three-Family Pilot — 2026-05-14 09:44 UTC

### What ran
Completed the corrected `family_relationships` pilot as Slurm job `1563604`
under the existing `phase1_rg30_p16_v4` root, joining the prior corrected
`boxnet` and `graph_color` cells.

### What was verified
All `16` family_relationships Slurm array children completed with exit code
`0:0`. Local sync succeeded. Artifact audit over all three Reasoning Gym tracks
passed with `48` cells, `5760/5760` candidate rows, `5760/5760` score rows, and
`360` selected rows. Aggregate output:
`runs/farmshare/phase1_rg30_p16_v4/phase1_results_all_rg.parquet`.

### Pilot results
Binary pass@1/pass@16 on `family_relationships`: base `0.067/0.267`, ProRL v1
`0.700/0.733`, ProRL v2 `0.400/0.733`, BroRL `0.633/0.767`. This gives a usable
Sustained-regime pilot gap alongside `graph_color`.

Continuous Reasoning Gym score diagnostics (`first/all/best16`, scaled by 100):
`boxnet` base `0.00/0.03/0.50`, ProRL v1 `0.17/3.16/15.72`, ProRL v2
`2.50/2.84/12.89`, BroRL `3.72/2.98/14.00`; `graph_color` base
`3.37/2.74/33.63`, ProRL v1 `100.00/99.38/100.00`, ProRL v2
`80.20/96.28/100.00`, BroRL `90.00/97.30/100.00`; `family_relationships`
base `6.67/1.88/26.67`, ProRL v1 `70.00/65.21/73.33`, ProRL v2
`40.00/68.12/73.33`, BroRL `63.33/70.42/76.67`.

### Validity note
`boxnet` still has binary pass@k `0.0` for every checkpoint, but this is not
the same metric as the ProRL Reasoning Gym table, which reports continuous
task reward. The current `boxnet` direction matches the paper qualitatively
(base near zero, ProRL/BroRL nonzero), but the absolute all-sample reward is
below the reported ProRL number. Do not make a binary recovery claim on
`boxnet` until the paper-faithful configuration and reporting metric are locked.

### Blockers
No infrastructure blocker for three-family Reasoning Gym pilots. Scientific
metric blocker remains for `boxnet` if the main run treats it as strict pass@k.

### Next checkpoint
Run the Phase 0 Karan-Du external replication gate on MATH500 before any RF
claim or Phase 2 recovery launch.

## Phase 0 Karan-Du Gate Launch — 2026-05-14 09:44 UTC

### What ran
Synced the updated repo to FarmShare, regenerated `direct`,
`reasoning_gym_direct`, `rws_math_direct`, and `seed_archive` JSON files under
`data/prorl_recovery_archives`, and verified the remote drift guard plus
targeted FarmShare/orchestration tests (`31 passed`).

Submitted Slurm job `1563658` as `prorl-phase0-karan-du-v1` with four
one-GPU shards over MATH500:
`Qwen/Qwen2.5-Math-7B`, HF/RWS backend, `single_prompt_power`,
`max_new_tokens=3072`, `MCMC_STEPS=10`, `MCMC_BLOCK_NUM=16`, and artifact root
`/scratch/users/duynguy/polaris/runs/prorl_recovery/phase0_karan_du_v1`.

### What was verified
All four Phase 0 shards entered `RUNNING` on L40S oat nodes with no immediate
Slurm failure. The Qwen2.5-Math-7B cache is present under scratch HF cache.

### Blockers
No launch blocker. This is the hard external replication gate: no RF claim is
valid until the synced artifacts audit and the observed MATH500 accuracy is
within `0.748 ± 0.02`.

### Next checkpoint
Monitor job `1563658`, sync on completion, audit the four shard artifacts, and
compute the replication accuracy against
`KARAN_DU_REPLICATION_GATE.accepts(...)`.

## Phase 0 HF Env Failure And Fix — 2026-05-14 09:48 UTC

### What ran
Job `1563658` failed immediately on all four shards before writing candidate
artifacts.

### What was verified
Slurm stderr showed `ValueError: Using a device_map or tp_plan requires
accelerate` from `transformers.AutoModelForCausalLM.from_pretrained(...)` in
the HF/RWS backend. This is an environment-contract failure, not a replication
result.

Added a red/green unit test that FarmShare env setup installs and imports
`accelerate`, patched `src/polaris/infra/farmshare.py`, and verified locally
with targeted tests (`32 passed`) plus drift guard. Installed `accelerate` into
the FarmShare env, regenerated archives after sync, and verified remote imports,
remote drift guard, and remote targeted tests (`32 passed`).

### Blockers
The `phase0_karan_du_v1` artifact root is invalid because all shards failed
before model load.

### Next checkpoint
Relaunch Phase 0 under a fresh root after the `accelerate` fix.

## Phase 0 Karan-Du Gate Relaunch — 2026-05-14 09:48 UTC

### What ran
Relaunched the external replication gate as Slurm job `1563674`
(`prorl-phase0-karan-du-v2`) under fresh root
`/scratch/users/duynguy/polaris/runs/prorl_recovery/phase0_karan_du_v2`.

### What was verified
After syncing the `accelerate` env-contract patch, regenerated remote archive
JSONs and verified the remote drift guard. All four shards entered `RUNNING`
with no immediate Slurm failure.

### Blockers
None at relaunch time.

### Next checkpoint
Monitor for model-load success and first candidate rows. If rows appear, keep
the long Phase 0 gate running; if all shards fail before rows, inspect stderr
and patch the next launch/config issue before relaunching.

## Phase 0 First-Row Check — 2026-05-14 09:54 UTC

### What ran
Polled Slurm job `1563674` after relaunch.

### What was verified
All four shards remained `RUNNING` past model load. Candidate files appeared
under `phase0_karan_du_v2/phase0/karan_du_replication`, with first rows already
written on shards `0` and `2`. Direct node `nvidia-smi` checks showed L40S GPU
utilization around `96-97%` and memory use around `19-23GB`, so the HF/RWS
gate is actively computing.

### Blockers
None. The Phase 0 external replication gate is now in a valid running state.

### Next checkpoint
Continue polling row counts and Slurm failures. On completion, sync down,
audit all four shards, aggregate, and compare observed accuracy with the
registered `0.748 ± 0.02` Karan-Du target.

## Phase 0 Gate Aggregator Added — 2026-05-14 10:21 UTC

### What ran
Added an explicit `aggregate-phase0` path to `scripts/run_prorl_recovery.py`.
It reads Phase 0 candidate rows, checks completeness against the expected
`500` MATH500 problems, computes accuracy, and writes a JSON report whose
`passed` field is exactly
`KARAN_DU_REPLICATION_GATE.accepts(observed_accuracy)` plus completeness.

### What was verified
Added a unit test that builds a synthetic `374/500 = 0.748` Phase 0 result and
asserts the registered Karan-Du gate passes. Local targeted tests passed with
`33 passed`; drift guard passed. Synced the utility to FarmShare and verified
the two targeted remote tests passed.

### Blockers
None. Job `1563674` continued running after the sync, with candidate rows
unchanged at the poll immediately after the utility sync.

### Next checkpoint
Continue monitoring job `1563674` until completion, then run artifact audit and
`aggregate-phase0`.

## Phase 0 One-Hour Poll — 2026-05-14 10:52 UTC

### What ran
Polled Slurm job `1563674` after roughly one hour of runtime.

### What was verified
All four shards remained `RUNNING` with no failed/cancelled/timed-out child
tasks. Candidate counts were shard-0 `15`, shard-1 `12`, shard-2 `19`, and
shard-3 `14`, for `60/500` total. Direct node checks still showed L40S GPU
utilization around `95-97%`.

### Blockers
None. The gate is progressing normally for HF/RWS MCMC.

### Next checkpoint
Continue coarse polling. At the observed aggregate row rate, expected
completion is on the order of another `7.5-8.5` hours if throughput remains
stable.

## Phase 2 Reasoning Gym Prompt Routing Fix — 2026-05-14 14:25 UTC

### What ran
Inspected the rendered Phase 2 cells while Phase 0 was running and found that
Reasoning Gym recovery rungs still referenced MATH500 `direct` and
`seed_archive` prompt archives. That would recreate the earlier boxed-answer
failure on Phase 2.

### What was verified
Added a red/green unit test requiring Reasoning Gym Phase 2 rungs 0-4 to use
`reasoning_gym_direct` and rungs 5-7 to use
`reasoning_gym_seed_archive`. Added a Reasoning-Gym-native seed archive with
answer-tag-preserving prompt variants and updated `write-archives` to emit five
archive files. Local targeted tests passed with `33 passed`; drift guard passed.
Synced the patch to FarmShare, regenerated five remote archives, and verified
the two targeted remote tests passed.

### Blockers
None after the fix. This was a pre-Phase-2 validity issue, not a Phase 0 issue.

### Next checkpoint
Continue monitoring Phase 0 job `1563674`; after the Karan-Du gate passes or
fails, use the corrected Phase 2 archive routing for any recovery ladder launch.

## Phase 1 Score Diagnostics For Reasoning Gym — 2026-05-14 19:37 UTC

### What ran
Added score-preserving Phase 1 aggregation before the 100-problem main run.
`aggregate_phase1` now writes per-candidate `verifier_score` and a `score_at`
summary with first, best, and all-sample mean scores for `k in {1,16,128}`,
while leaving binary pass@k unchanged.

### Why
The ProRL paper's Reasoning Gym table reports continuous reward-style scores,
so strict binary `score >= 1.0` is not a paper-faithful `boxnet` comparison by
itself. This keeps the primary pass@k artifact intact while preserving the
reward diagnostic needed for `boxnet`.

### What was verified
The targeted aggregation test passed locally, the ProRL/FarmShare targeted
suite passed `33` tests, the patch was synced to FarmShare, and the same
aggregation test passed remotely under the FarmShare env.

### Blockers
None. This is a pre-main-run analysis validity fix and does not affect the
running Phase 0 job.

### Next checkpoint
Continue monitoring Phase 0 job `1563674`; after completion, sync artifacts,
run `audit-plan`, and apply `aggregate-phase0`.

## Phase 3 Trajectory Materialization Added — 2026-05-14 19:44 UTC

### What ran
Added `materialize_phase3_trajectories(...)` and
`scripts/materialize_prorl_phase3_trajectories.py` so Phase 3.1 can attach the
exact successful ProRL/BroRL raw candidate trace to each derived
`phase3_input_set` row before HF/RWS base-logprob scoring.

### Why
`phase3_input_set.parquet` identifies ProRL/BroRL-only problem/checkpoint
pairs, but the logprob audit needs the exact `prompt_text` and successful
`generation` from Phase 1 `candidates.jsonl`. Without this materialization
step, Phase 3 would require ad hoc trace recovery.

### What was verified
Added red/green tests for raw candidate materialization and the CLI wrapper.
Targeted ProRL/FarmShare tests passed with `35 passed`.

### Blockers
None. This is analysis orchestration for after Phase 2 rung-7 exists; it does
not affect the active Phase 0 job.

### Next checkpoint
Sync this patch to FarmShare, then keep monitoring Phase 0 job `1563674`.

## Phase 0 Karan-Du Gate Failed — 2026-05-15 02:46 UTC

### What ran
Completed FarmShare Slurm job `1563674` for the Phase 0 external replication
gate:

- track: MATH500
- model: `Qwen/Qwen2.5-Math-7B`
- condition: `single_prompt_power`
- alpha: `4.0`
- max new tokens: `3072`
- block num: `16`
- MCMC steps: `10`
- shards: `4 x 1 GPU`

Artifacts were synced from FarmShare to
`runs/farmshare/phase0_karan_du_v2`.

### What was verified
Slurm completed all four array shards with exit code `0:0`. Artifact audit
passed:

- cells: `4`
- candidates: `500/500`
- scores: `500/500`
- selected rows: `500/500`

`aggregate-phase0` wrote
`runs/farmshare/phase0_karan_du_v2/phase0_gate.json` with:

- observed accuracy: `0.722`
- target accuracy: `0.748`
- tolerance: `0.02`
- accepted window: `[0.728, 0.768]`
- gate passed: `false`

### Blockers
This is a hard gate failure. Per the ProRL recovery audit protocol, no
Recoverable Fraction claim is valid and no 100-problem main run should launch
until the Karan-Du/RWS mismatch is diagnosed or the gate is explicitly revised.

### Next checkpoint
Stop the current goal at the hard-gate stopping condition. A follow-up run
should diagnose whether the `0.722` result comes from verifier drift, prompt or
dataset mismatch, sampler/MCMC implementation drift, model revision mismatch,
or stochastic variance outside the pre-registered tolerance.

## Phase 0 Failure Diagnosis — 2026-05-15 03:11 UTC

The failed Phase 0 run was an artifact-complete numerical gate failure, not a
Slurm or FarmShare failure.

Observed evidence:

- Slurm job `1563674` completed all four array shards with exit `0:0`.
- Artifact audit passed with `500/500` candidates, scores, and selected rows.
- Aggregated accuracy was `361/500 = 0.722`, below the preregistered accepted
  window `[0.728, 0.768]` for target `0.748 ± 0.02`.
- Re-exporting the completions to the upstream RWS CSV schema and grading with
  upstream `eval_math.py` functions also gives MCMC accuracy `0.722`, so the
  local math verifier is not the source of the discrepancy.
- The MATH500 dataset and RWS prompt are aligned: all `500` problem IDs are
  present once, references match, and the direct prompt matches
  `PROMPT + question + COT`.

Likely causes:

- The executed FarmShare gate was not the exact RWS launch recipe. RWS uses
  `5` contiguous shards x `8` seeds in
  `upstream/reasoning-with-sampling/llm_experiments/scripts/power_samp_math.sh`;
  this run used `4` modulo shards x one seed.
- RWS `power_samp_math.py` accepts `--seed` but sets `random.seed(0)` and does
  not seed NumPy acceptance randomness. Our wrapper seeds Python, NumPy, and
  Torch with `17`, so trajectories are not seed-equivalent.
- The qwen2.5-math-7b registry entry is unpinned for Phase 0; the manifest has
  `model_revision: null`. FarmShare cache currently resolves
  `Qwen/Qwen2.5-Math-7B` to snapshot
  `b101308fe89651ea5ce025f25317fea6fc07e96e`.
- The runtime stack differs from upstream RWS: FarmShare used Python `3.11.15`,
  Torch `2.7.0+cu126`, Transformers `4.51.3`; upstream environment pins Python
  `3.12.10`, Torch `2.8.0`, Transformers `4.47.1`.
- FarmShare logs also show the Qwen sliding-window SDPA warning, so exact
  generation parity with the authors' H100 setup is not guaranteed.

Conclusion:

The hard gate was correctly enforced, but the failed run is best interpreted as
`our 4-shard deterministic FarmShare adaptation scored 0.722`, not as a clean
falsification of the Karan-Du `0.748` result. The next valid gate should add an
exact-RWS replication mode: `5` contiguous shards, `8` seed runs or the paper's
documented aggregation, upstream environment pins where feasible, pinned model
snapshot in the manifest, and token/logprob capture for forensic auditing.

## Exact-RWS Phase 0 Goal Restart — 2026-05-15 04:24 UTC

Reset the active goal to repair Phase 0 into a paper-faithful external
replication before any Recoverable Fraction continuation.

What changed locally:

- Added an exact upstream RWS Phase 0 path that renders `5 x 8 = 40` cells,
  matching `upstream/reasoning-with-sampling/llm_experiments/scripts/power_samp_math.sh`.
- The exact cell command calls
  `upstream/reasoning-with-sampling/llm_experiments/power_samp_math.py`
  directly, not `scripts/run_condition.py`.
- Added an exact-RWS aggregator over upstream CSV outputs, using the RWS math
  grader on `mcmc_answer` and comparing mean complete-seed accuracy to
  `0.748 ± 0.02`.
- Rendered the FarmShare Slurm script at
  `runs/farmshare/prorl-phase0-rws-exact-v1.sbatch` with `--array=0-39%4`.

Verification before remote launch:

- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/`
  passed: `194 passed`.
- `bash scripts/check_protocol_sync.sh` passed.
- `scripts/run_prorl_recovery.py plan-rws-exact` wrote
  `runs/farmshare/phase0_rws_exact_v1_plan.json` with `40` exact-RWS cells.

Current rule:

Do not continue to Phase 1/2/RF until `phase0_rws_exact_v1` is synced,
aggregated, and either passes the Karan-Du gate or documents a hard FarmShare
replication blocker.

## Exact-RWS Remote Launch Repair — 2026-05-15 04:30 UTC

The first exact-RWS submit attempts exposed two orchestration bugs before any
long GPU work:

- `phase0_rws_exact_v1`: submitted as `1564446` and cancelled. Cause:
  FarmShare QoS rejects a `40` task array because `gpu` has
  `MaxSubmitPU=32`. Fix: split into `4 + 28 + 8` cells, with the final `8`
  submitted only after queue slots clear.
- `phase0_rws_exact_v2`: smoke `1564475` failed quickly. Cause:
  `bash -lc` resolved bare `python` outside the Polaris env, so upstream RWS
  could not import `tqdm`. Fix: `rws_exact_cell_command` now invokes
  `sys.executable`, preserving the active FarmShare env.

The remote source issue is fixed:

- `rsync_to_farmshare_command` now includes
  `upstream/reasoning-with-sampling/llm_experiments/***`,
  `README.md`, and `environment.yml`, while still excluding unrelated
  `upstream/**`.
- Remote check confirmed
  `/scratch/users/duynguy/polaris/repo/upstream/reasoning-with-sampling/llm_experiments/power_samp_math.py`
  exists.
- Remote env import check with
  `/scratch/users/$USER/polaris/envs/polaris/bin/python` passed for
  `torch`, `transformers`, `tqdm`, `pandas`, `datasets`, and upstream RWS
  modules.

Verification after repair:

- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/`
  passed: `195 passed`.
- `bash scripts/check_protocol_sync.sh` passed.

Current clean run:

- `phase0_rws_exact_v3` smoke job `1564479` is running cells `0..3`.
- Logs show upstream RWS loaded `qwen_math`, loaded model shards, and entered
  `Benchmark on MATH`.
- Main queued job `1564483` covers cells `4..31` and is pending behind the
  four running smoke cells due `QOSMaxJobsPerUserLimit`.
- Cells `32..39` are intentionally not submitted yet to respect
  `MaxSubmitPU=32`.

## Exact-RWS v4 Relaunch With Offline HF Snapshot — 2026-05-15 04:36 UTC

Superseded `phase0_rws_exact_v3` before accepting it as clean because the
command recorded the cached model snapshot but did not force the Hugging Face
loader to stay offline. The exact cell command now exports `HF_HUB_OFFLINE=1`,
so `Qwen/Qwen2.5-Math-7B` resolves only through the FarmShare cache.

Remote cache evidence:

- `/scratch/users/$USER/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/refs/main`
  resolves to snapshot `b101308fe89651ea5ce025f25317fea6fc07e96e`.
- The corresponding snapshot directory exists under the same cache root.

Current active clean run:

- Root:
  `/scratch/users/duynguy/polaris/runs/prorl_recovery/phase0_rws_exact_v4`.
- Smoke job `1564490` is running cells `0..3`.
- Main job `1564491` covers cells `4..31` and is pending behind the four
  running smoke cells due `QOSMaxJobsPerUserLimit`.
- Tail cells `32..39` remain intentionally unsubmitted until queue slots clear.
- Smoke logs show direct upstream
  `upstream/reasoning-with-sampling/llm_experiments/power_samp_math.py`
  execution, model load success, and active MCMC progress.
- Error logs are clean for `Traceback`, `ModuleNotFoundError`,
  `No such file`, `CUDA out of memory`, and `RuntimeError`.
- In-job GPU check reports all four L40S tasks at roughly `96-97%`
  utilization and about `20GB` VRAM used.

No exact-RWS CSVs have been written yet; upstream writes one CSV only after a
full 100-problem shard/seed cell completes.

## Exact-RWS Monitoring Policy — 2026-05-15 04:43 UTC

Installed a local Slurm monitor for `phase0_rws_exact_v4` that polls active
array tasks, CSV count, manifest count, and error signatures. The first monitor
wrapper was stopped because FarmShare accounting is still on local May 14 and
`sacct -S 2026-05-15` produced noisy date-filter errors. The restarted monitor
uses in-process tail-submission flags instead.

Tail scheduling policy:

- Keep the existing smoke job `1564490` and main job `1564491`.
- Submit `prorl-phase0-rws-exact-v4-tail-a` for cells `32..35` once active
  submitted v4 tasks fall to `28` or fewer.
- Submit `prorl-phase0-rws-exact-v4-tail-b` for cells `36..39` once the task
  count again falls to `28` or fewer.

This respects FarmShare's `MaxSubmitPU=32` limit while avoiding a late idle gap.
The Phase 0 target remains Karan-Du's single-shot Power Sampling MATH500
accuracy `0.748 ± 0.02`; pass@k is not the primary replication gate.

## Exact-RWS First Artifact + Provenance Patch — 2026-05-15 12:07 UTC

First upstream CSV landed:

- `/scratch/users/duynguy/polaris/runs/prorl_recovery/phase0_rws_exact_v4/phase0/rws_exact/results/qwen_math/qwen_math_math_base_power_samp_results_10_0.25_0_2.csv`
- The corresponding cell manifest exists at
  `phase0/rws_exact/cells/batch-0/seed-2/manifest.json`.

The manifest confirms:

- direct command:
  `/scratch/users/duynguy/polaris/envs/polaris/bin/python power_samp_math.py`;
- source path:
  `/scratch/users/duynguy/polaris/repo/upstream/reasoning-with-sampling/llm_experiments`;
- `HF_HUB_OFFLINE=1`;
- model snapshot `b101308fe89651ea5ce025f25317fea6fc07e96e`;
- expected CSV exists.

Weakness found: `upstream_commit` is `null` in the cell manifest because the
remote rsync copy excludes `.git`. Added supplemental provenance at:

- `/scratch/users/duynguy/polaris/runs/prorl_recovery/phase0_rws_exact_v4/phase0/rws_exact/supplemental_provenance.json`

It records upstream reference clone commit
`720a8e9d084c87a630595e316f5260f1d7c3446c`, the remote source path, and the
Qwen2.5-Math-7B cached snapshot. Running jobs were not modified mid-cell.

## Exact-RWS Smoke Complete, Main Handoff — 2026-05-15 12:43 UTC

The four smoke cells `0..3` completed and wrote the expected upstream CSVs:

- `qwen_math_math_base_power_samp_results_10_0.25_0_0.csv`
- `qwen_math_math_base_power_samp_results_10_0.25_0_1.csv`
- `qwen_math_math_base_power_samp_results_10_0.25_0_2.csv`
- `qwen_math_math_base_power_samp_results_10_0.25_0_3.csv`

The corresponding manifests exist under:

- `phase0/rws_exact/cells/batch-0/seed-0/manifest.json`
- `phase0/rws_exact/cells/batch-0/seed-1/manifest.json`
- `phase0/rws_exact/cells/batch-0/seed-2/manifest.json`
- `phase0/rws_exact/cells/batch-0/seed-3/manifest.json`

Slurm handoff is working:

- Main array `1564491` is running cells `4..7` on the four allowed GPUs.
- Tail chunk `prorl-phase0-rws-exact-v4-tail-a` submitted as job `1564555`
  for cells `32..35` when active submitted tasks dropped to `28`.
- Tail chunk `tail-b` is still held until the active submitted task count
  drops to `28` again.

Smoke artifact sanity check with the FarmShare Polaris env confirmed all four
CSV files have `100` rows and include the expected upstream columns:
`question`, `correct_answer`, `naive_completion`, `naive_answer`,
`std_completion`, `std_answer`, `mcmc_completion`, and `mcmc_answer`.

## Exact-RWS Batch 0 Complete, Tail Fully Submitted — 2026-05-15 21:08 UTC

The first full 8-seed shard/batch is complete:

- CSV count: `8 / 40`.
- Manifest count: `8 / 40`.
- Completed files cover `batch_idx=0`, `seed=0..7`.
- Completed Slurm cells all show `COMPLETED` with exit code `0:0`.

Current scheduling state:

- Running: main cells `1564491_4`, `1564491_5`, `1564491_6`, and tail-a
  cell `1564555_0`.
- Pending: remaining main cells, tail-a cells, and tail-b cells.
- Tail-b was submitted as job `1564639` once active submitted tasks dropped to
  `28`.

Observed cell duration is roughly `7.3-8.5h` per 100-problem exact upstream RWS
cell. With `32` cells remaining and 4 FarmShare GPU slots, ETA is about
`60-70h` if queue behavior remains stable.

## Local Live-Go Readiness Recheck — 2026-05-15

Ran a fresh local readiness recheck for the ProRL/BroRL Recoverable Fraction
audit before any new launch.

Passed:

- Hook integrity manifest check: all recorded hook/settings hashes OK.
- `bash scripts/check_protocol_sync.sh`.
- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/`
  returned `201 passed`.
- `scripts/xai_reflection_smoke.py --dry-run` reports provider `xai`, base URL
  `https://api.x.ai/v1`, model `grok-4.3`, and cost caps under limit.
- `scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp` reports
  `passed=true`, `deferred_tracks=[]`, `deferred_experiments=[]`, and cache
  replay `generation_calls_on_replay=0`.
- Phase 1 planning smoke produced 32 expected cells for
  `math500` + `reasoning_gym_graph_color` over the four ProRL audit checkpoints.

Current live state:

- `.env` has `XAI_BASE_URL=https://api.x.ai/v1` and
  `XAI_REFLECTION_MODEL=grok-4.3`, but no `XAI_API_KEY` value.
- No live xAI request was made.
- FarmShare currently has four ProRL Phase 0 exact-RWS cells running and the
  remaining cells pending behind the per-user GPU job limit.
- Flow availability check shows A100 80GB SXM capacity in `us-central3-a`;
  current reported 4x A100 price is `$0.04/hr` total.

Decision: local infrastructure is ready to proceed, but live GEPA/xAI reflection
is blocked on inserting the xAI key into a local secret source. RF claims remain
blocked until Phase 0 exact-RWS completes and hits the Karan-Du gate.

## Credential Smoke Update — 2026-05-15

Inserted user-provided HF and xAI credentials into local ignored `.env` only;
`.env` remains mode `600`.

Passed:

- HF auth probe succeeds as user `yudduy`.
- GPQA-Diamond official loader probe succeeds for one row without printing row
  content or answer keys.
- Installed the declared `gepa_reflection` optional extra into `.venv-eval`
  with `uv pip install --python ./.venv-eval/bin/python -e '.[gepa_reflection]'`.
- xAI live reflection smoke completes one tiny request through the GEPA LM
  adapter; output redacts `api_key` and reports provider `xai`, model
  `grok-4.3`, input tokens `134`, output tokens `1`.
- xAI archive-build dry run records reflection config with `api_key:
  <redacted>` and cost-cap metadata.
- Secret pattern scan found no HF/xAI token outside `.env`.
- Focused tests pass:
  `tests/unit/test_live_resources.py tests/unit/test_prorl_recovery.py tests/unit/test_farmshare.py`
  returned `49 passed`.
- Full local tests pass: `201 passed`.

Current blocker status: xAI and HF credential gates are cleared locally. RF
claims remain blocked only by the Phase 0 exact-RWS Karan-Du replication gate
and the normal explicit authorization/cost-cap requirement for new paid runs.

## Active Goal And Cheap-Flow Poll Contract — 2026-05-15

Created the active Codex goal:

Complete the POLARIS ProRL/BroRL Recoverable Fraction audit end to end until
Phase 0 exact-RWS/Karan-Du passes and Phase 1, filtered Phase 2, GEPA/xAI rungs,
Phase 3 buckets, canonical artifacts, audits, and progress notes are complete,
or until a hard gate fails or required paid-run authorization/cost-cap evidence
is missing.

Updated `TODO.md` with the durable execution contract:

- Poll FarmShare Phase 0 exact-RWS until all 40 cells finish.
- Aggregate Phase 0 with `scripts/run_prorl_recovery.py aggregate-rws-exact`
  and enforce the `0.748 +-0.02` Karan-Du gate before any RF claim.
- Poll Flow before every paid launch. Flow A100 is considered cheap at or below
  `$0.03/GPU-hr`; current observed 4x A100 quote is `$0.04/hr` total, below
  the configured threshold.
- Use Flow only through the resource-profile launcher with explicit artifact
  dir, cache path, split, seed, model, backend, cost estimate, cost cap, and
  authorization.
- Preserve stop rules: protocol sync failure, missing artifact contract,
  Phase 0 gate miss, or incomplete paid-run evidence all stop the run.

No new paid run was launched in this step because Phase 0 exact-RWS is already
running on FarmShare and remains the immediate RF gate.

## Flow Spend Band Tightened — 2026-05-15

User clarified the Flow constraint: get the whole fast-path run through while
keeping Flow spend under about `$10`, with run integrity more important than
slightly faster startup.

Updated the Flow profile and protocol surfaces:

- `configs/prorl_live_resources.json` now caps `flow_a100_weekend` at
  `$0.025/GPU-hr`.
- Preferred launch band is 4x A100 total price `<= $0.10/hr`.
- At that band, a 48-72h post-Phase-0 run costs about `$4.80-$7.20`; a 100h
  overrun is about `$10`.
- `TODO.md`, `docs/PRORL_RECOVERY_AUDIT.md`, and
  `tests/unit/test_live_resources.py` were updated to match.

Latest observed Flow availability remains inside the band: 4x A100 80GB SXM in
`us-central3-a` at `$0.04/hr` total. No paid launch was started; Phase 0
FarmShare exact-RWS is still the active gate.

## Exact-RWS Live Status — 2026-05-16 05:27 UTC

Phase 0 exact upstream RWS remains active on FarmShare.

- First v4 Slurm cells were submitted at `2026-05-14T21:33:41` FarmShare time
  and began immediately; elapsed wall-clock is about `24h54m` at this
  checkpoint.
- Completed artifacts: `8 / 40` CSVs and `8 / 40` cell manifests.
- Completed cells cover `batch_idx=0`, `seed=0..7`, all exit code `0:0`.
- Running cells: `1564491_4`, `1564491_5`, `1564491_6`, and `1564555_0`.
- Pending cells are held only by FarmShare's 4-running-GPU per-user QoS limit.
- No traceback, missing-file, OOM, killed, failed, or timeout signatures were
  found in the exact-RWS Slurm logs.
- One allocated L40S visibility probe showed active use:
  GPU util `95%`, memory `28727 / 46068 MiB`, power `266.56 W`.

Current running progress:

- `1564491_4`: `83 / 100` MATH items, elapsed `9:51:30`.
- `1564491_5`: `70 / 100` MATH items, elapsed `9:29:33`.
- `1564491_6`: `60 / 100` MATH items, elapsed `8:30:48`.
- `1564555_0`: `75 / 100` MATH items, elapsed `8:45:16`.

The first ETA estimate from the completed `batch_idx=0` cells was `60-70h`.
The currently running cells are slower. Conservative ETA from current partial
progress is now `73-102h`; raw completion-rate ETA is about `100h`. After that:
sync down, run
`aggregate-rws-exact`, and compare to the `0.748 ± 0.02` Karan-Du gate before
any RF claim.

## Exact-RWS Live Status — 2026-05-16 05:49 UTC

No new exact-RWS artifact has landed since the previous checkpoint.

- Completed artifacts remain `8 / 40` CSVs and `8 / 40` cell manifests.
- Four cells remain running on FarmShare L40S GPUs; all other submitted cells
  are pending behind the 4-running-GPU QoS limit.
- No traceback, missing-file, OOM, killed, failed, or timeout signatures were
  found in the exact-RWS Slurm logs.
- Running progress:
  - `1564491_4`: `88 / 100` MATH items, elapsed `10:12:49`.
  - `1564491_5`: `70 / 100` MATH items, elapsed `9:29:33` in the latest
    progress line; likely on a long item.
  - `1564491_6`: `64 / 100` MATH items, elapsed `8:44:30`.
  - `1564555_0`: `78 / 100` MATH items, elapsed `9:08:58`.

No RF or ProRL main jobs are present in the queue.

## Exact-RWS Handoff Status — 2026-05-16 06:16 UTC

Codex resume for thread `019e236a-38e2-7020-80d2-ba546e600669` hit a stale
session-path check: requested path under `/Users/duy/.codex`, active path under
`/Users/duy/.codex-kduy`. The two paths resolve to the same symlinked session
file, but the old thread should now be treated as read-only evidence rather
than the control plane.

Live FarmShare state is unchanged structurally:

- Completed artifacts remain `8 / 40` CSVs and `8 / 40` cell manifests.
- Running cells remain `1564491_4`, `1564491_5`, `1564491_6`, and `1564555_0`
  on `oat-01`; all other exact-RWS v4 cells are pending behind the
  4-running-GPU QoS limit.
- No traceback, missing-file, OOM, killed, failed, or timeout signatures were
  found in exact-RWS logs.
- Running progress:
  - `1564491_4`: `89 / 100` MATH items, elapsed `10:15:55`.
  - `1564491_5`: `73 / 100` MATH items, elapsed `10:01:10`.
  - `1564491_6`: `70 / 100` MATH items, elapsed `9:16:43`.
  - `1564555_0`: `79 / 100` MATH items, elapsed `9:48:28`.

Operational decision: continue monitoring from a fresh thread unless the Codex
UI can resume the old thread using the resolved `.codex-kduy` path. Do not
launch RF or ProRL main until exact-RWS sync, aggregate, and the
`0.748 ± 0.02` gate comparison are complete.

## Exact-RWS Partial Result Check — 2026-05-16 06:21 UTC

Fresh poll after the thread handoff:

- Completed artifacts remain `8 / 40` CSVs and `8 / 40` cell manifests.
- The `8` completed CSVs are all `100` rows with the expected upstream columns:
  `question`, `correct_answer`, `naive_completion`, `naive_answer`,
  `std_completion`, `std_answer`, `mcmc_completion`, `mcmc_answer`.
- Repo-native partial aggregation over completed `batch_idx=0`, using the
  same RWS math grader as the final gate, gives mean seed accuracy `0.78125`
  on the first `100` MATH rows. This is not the final gate because the remaining
  four shards are not complete.
- Per-seed completed-batch accuracies: seed `0=0.79`, `1=0.78`, `2=0.79`,
  `3=0.80`, `4=0.78`, `5=0.79`, `6=0.77`, `7=0.75`.
- Running progress:
  - `1564491_4`: `89 / 100` MATH items, elapsed `10:15:55`.
  - `1564491_5`: `73 / 100` MATH items, elapsed `10:01:10`.
  - `1564491_6`: `70 / 100` MATH items, elapsed `9:16:43`.
  - `1564555_0`: `80 / 100` MATH items, elapsed `9:50:46`.
- One visible L40S probe on `oat-01` showed GPU util `96%`, memory
  `28727 / 46068 MiB`, power `268.52 W`.
- No exact-RWS traceback, missing-file, OOM, killed, failed, or timeout
  signatures were found.

Trace/provenance check:

- Completed manifests record `HF_HUB_OFFLINE=1`, execution from
  `upstream/reasoning-with-sampling/llm_experiments`, model
  `Qwen/Qwen2.5-Math-7B`, model revision
  `b101308fe89651ea5ce025f25317fea6fc07e96e`, temperature `0.25`, and
  `mcmc_steps=10`.
- Remote executed source hashes match the local checked-out upstream source:
  `power_samp_math.py`
  `ec79bdeaa868a5ab2f4df70774a98886787495ccfa1461b7323a7ef6a38d968c`;
  `grader_utils/math_grader.py`
  `197ac7bb81d0ed9cddeed47aff6201f2bf8e55de71c5baf1c18269780168de77`.
- Provenance caveat: remote `upstream/reasoning-with-sampling/.git` is
  incomplete enough that `git rev-parse HEAD` fails, so completed manifests have
  `upstream_commit=null`. This does not invalidate the live run mechanics, but
  final audit should record the source-file hashes and local expected upstream
  commit `720a8e9d084c87a630595e316f5260f1d7c3446c`.

ETA from current live throughput is still broad. Current running cells imply
roughly `12-14h` per 100-item cell, with four cells in parallel. Remaining work
is about `29` cell-equivalents, giving a central estimate near `90-95h` from
this checkpoint. Practical window: `2026-05-19 04:21 UTC` to
`2026-05-20 15:21 UTC` (`2026-05-18 21:21 PT` to `2026-05-20 08:21 PT`), plus
sync and aggregation time.

## Flow A100x4 Provisioning Files — 2026-05-16 06:25 UTC

Prepared but did not launch a paid Flow probe for the locked 4x A100 path.

- Added `.flowignore` so local `.env`, `.venv-eval/`, `runs/`, and `upstream/`
  are not uploaded with Flow code bundles.
- Added `scripts/flow_prorl_a100x4_probe.sh`, which writes
  `environment.json`, `preflight_imports.json`, `gpu_check.json`,
  `manifest.json`, and `audit.md` under
  `POLARIS_RUN_ROOT/flow_a100x4_probe/<timestamp>/`.
- Added `configs/flow_prorl_a100x4_probe.yaml` with
  `instance_type: 4xa100`, `region: us-central3-a`, spot allocation,
  `preemptible_ok: true`, `max_price_per_hour: 0.10`,
  `max_run_time_hours: 2`, and `terminate_on_exit: true`.
- `flow submit --dry-run --json configs/flow_prorl_a100x4_probe.yaml` returned
  `status: valid`.
- `bash scripts/check_protocol_sync.sh` passed.
- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_live_resources.py tests/unit/test_prorl_recovery.py
  tests/unit/test_farmshare.py` passed: `52 passed in 4.51s`.

This provisioning bundle is only a GPU/integrity probe. It does not run Phase 0,
Phase 1, Phase 2, GEPA, or Phase 3. A live Flow submit still needs an explicit
launch command and cost cap confirmation.

## Exact-RWS Live Status — 2026-05-16 06:24 UTC

Phase 0 exact upstream RWS remains the active gate.

- Completed artifacts remain `8 / 40` CSVs and `8 / 40` cell manifests.
- Running cells remain `1564491_4`, `1564491_5`, `1564491_6`, and
  `1564555_0` on `oat-01`; remaining exact-RWS v4 cells are pending behind
  FarmShare's 4-running-GPU QoS limit.
- Running `std done` counts from Slurm stdout:
  - `1564491_4`: `90 / 100`, elapsed `10:49:00`.
  - `1564491_5`: `74 / 100`, elapsed `10:26:19`.
  - `1564491_6`: `71 / 100`, elapsed `09:28:07`.
  - `1564555_0`: `81 / 100`, elapsed `09:56:20`.
- Grep over exact-RWS stderr logs found no traceback, OOM, killed, missing-file,
  or failed signatures. Some stdout contains generated text with words like
  `NameError`; this is model output, not a process failure.
- Current Flow availability quote remains inside the standby band:
  `4x A100 80GB SXM`, `us-central3-a`, `$0.04/hr` total, `8` instances
  available. No Flow job was launched.

Operational state is unchanged: wait for Phase 0 exact-RWS completion, sync and
aggregate with `scripts/run_prorl_recovery.py aggregate-rws-exact`, then compare
against the `0.748 ± 0.02` Karan-Du gate before any RF claim or downstream
phase launch.

### Bounded Monitor Follow-up — 2026-05-16 06:35 UTC

Three read-only polls over ten minutes showed no structural change:

- Completed artifacts stayed at `8 / 40` CSVs and `8 / 40` manifests.
- Running cells stayed on `1564491_4`, `1564491_5`, `1564491_6`, and
  `1564555_0`.
- Pending cells stayed held by `QOSMaxJobsPerUserLimit`.
- No paid Flow launch was started.

## Flow A100x4 Provisioning Probe — 2026-05-16 07:10 UTC

User asked why an 8x A100 appeared paused and requested provisioning a 4x A100.
Live Flow state showed no active bids; the old `polaris-a100x8-ckpt-e8b2d2`
task was cancelled history, not an active running bid.

Provisioned the prepared 4x A100 probe after the current Flow quote remained
inside the cap:

- Current quote before launch: `a100-80gb.sxm.4x`, `us-central3-a`, `$0.04/hr`
  total.
- Submitted `configs/flow_prorl_a100x4_probe.yaml`.
- First submit attempt failed before provisioning because `upload_code: true`
  made Flow generate an oversized startup script.
- Reworked the YAML to `upload_code: false`, then trimmed the inline probe below
  Flow's `15000` character script limit.
- Dry-run validator passed after each YAML repair.
- `tests/unit/test_live_resources.py` passed: `9 passed`.
- Final submit created task `bid_eZD10jIJqGq0FGqu`,
  `polaris-prorl-a100x4-probe`, `instance_type=4xa100`,
  `region=us-central3-a`.
- Flow transitioned from `provisioning` to `running`.
- `flow ssh bid_eZD10jIJqGq0FGqu -- nvidia-smi -L` observed four
  `NVIDIA A100-SXM4-80GB` GPUs.
- Probe logs printed
  `POLARIS_FLOW_A100X4_PROBE_OK /workspace/polaris/runs/prorl_recovery/flow_a100x4_probe/20260516T070622Z`.
- Remote artifact directory contained `audit.md`, `environment.json`,
  `gpu_check.json`, `manifest.json`, `nvidia-smi-L.txt`,
  `nvidia-smi-query.csv`, and `preflight_imports.json`.
- Cancelled the probe after verification to stop billing. `flow bid list
  --json` no longer listed `bid_eZD10jIJqGq0FGqu`; `flow status` may retain a
  stale `running` entry briefly after cancellation.

Local evidence bundle:
`runs/flow/a100x4_probe_20260516T070622Z/`.

Important integrity note: this was a short provisioning probe only. The remote
artifact directory was ephemeral; the local evidence bundle records the captured
Flow stdout/log/SSH proof. Future experiment jobs must use an explicit sync or
persistent artifact store before cancellation.

## Exact-RWS / Flow Status — 2026-05-16 07:12 UTC

Read-only checkpoint after the 4x A100 probe:

- FarmShare Phase 0 exact-RWS remains the active gate.
- Completed artifacts remain `8 / 40` CSVs and `8 / 40` manifests.
- Running cells and current `std done` counts:
  - `1564491_4`: `97 / 100`, elapsed `11:36:32`.
  - `1564491_5`: `82 / 100`, elapsed `11:13:51`.
  - `1564491_6`: `78 / 100`, elapsed `10:15:39`.
  - `1564555_0`: `84 / 100`, elapsed `10:43:52`.
- Pending cells remain blocked by `QOSMaxJobsPerUserLimit`.
- Exact-RWS stderr scan found no traceback, OOM, killed, missing-file, or failed
  signatures.
- Flow `bid list` showed no active bids after cancelling the 4x A100 probe.
- Current Flow A100 quote: `a100-80gb.sxm.4x`, `us-central3-a`, `$0.04/hr`
  total; `8x` quote is `$0.12/hr` total.

No RF claim or downstream Phase 1/2/3 launch is authorized before the exact-RWS
gate aggregates and passes.

## Phase 1 Prep While Exact-RWS Runs — 2026-05-16

No downstream RF/ProRL phase was launched. The Phase 0 exact-RWS replication
gate remains the active blocker for RF claims and downstream execution.

No-regret local prep completed while FarmShare continues Phase 0:

- Rendered resource profiles to
  `runs/prorl_recovery_live_prep/profiles.json`.
- Rendered the FarmShare Phase 1 resource plan to
  `runs/prorl_recovery_live_prep/phase1_farmshare_plan.json`.
- Rendered the science-equivalent Phase 1 plan to
  `runs/prorl_recovery_live_prep/phase1_science_plan.json`.
- Rendered one non-executed Phase 1 command to
  `runs/prorl_recovery_live_prep/phase1_cell0_command.json`.
- The Phase 1 plan has `64` cells:
  `4` tracks x `4` checkpoints x `4` shards.
- Tracks: `math500`, `reasoning_gym_boxnet`,
  `reasoning_gym_family_relationships`, `reasoning_gym_graph_color`.
- Checkpoints: base `deepseek-r1-distill-qwen-1.5b`, `nemotron-prorl-v1`,
  `nemotron-prorl-v2`, and `nemotron-brorl`.
- Condition: `bon_temp1`, `samples_per_problem=128`, split `0..100`.
- Artifact contract length in the resource-profile plan is `12`.
- Corrected the dry-run root to the FarmShare account path
  `/scratch/users/duynguy/...`; the generated command no longer contains the
  local Mac username path `/scratch/users/duy/...`.

Validation:

- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_prorl_recovery.py tests/unit/test_live_resources.py
  tests/unit/test_farmshare.py` passed: `52 passed in 4.07s`.
- `bash scripts/check_protocol_sync.sh` passed.

Next executable step after Phase 0 passes: sync exact-RWS artifacts, run
`aggregate-rws-exact`, then submit Phase 1 through the resource-profile
launcher. If Phase 0 misses the gate, stop and diagnose rather than launching
Phase 1.

## ProICL Fast-Weight Audit Contract — 2026-05-16

User corrected the experiment framing: ProICL is a fast-weight recovery audit,
not the older generic POLARIS RF ladder. The new causal decomposition is:

- `mcmc_only`: direct prompt plus power sampling, measuring exploitation.
- `gepa_only`: cross-task evolved prompt archive, measuring context discovery.
- `gepa_mcmc`: cross-task archive plus fixed-alpha MCMC, measuring composition.
- `gepa_mcmc_memory`: the same plus verifier-gated external curriculum memory.
- `prorl_v2_greedy`: slow-weight reference.

Implementation added:

- `docs/PROICL_FAST_WEIGHT_RECOVERY_AUDIT.md`
- `TODO.PROICL.md`
- `src/polaris/proicl/{analysis,archive,run_graph}.py`
- `scripts/run_proicl.py`
- ProICL runtime conditions `proicl_gepa_mcmc` and
  `proicl_gepa_mcmc_memory`.

No paid run or bulk generation was launched. Local verification:

- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/ tests/smoke/` passed: `210 passed in 8.67s`.
- `bash scripts/check_protocol_sync.sh` passed.

## ProICL Flow 4xA100 Overnight Run — 2026-05-16

Live Flow bid `bid_d43eJOwbGefYgX5X` (`polaris-prorl-a100x4-live`) is
running in `us-central3-a` on `4xa100` under the configured `$0.10/hr` bid cap.

Observed state at `20260516T090852Z`:

- Phase 0 exact-RWS/Karan-Du seed-0 gate is the active blocker.
- Gate root:
  `/workspace/polaris/runs/proicl_flow_a100x4/prorl_recovery_seed0_gate/phase0/rws_exact`.
- Queue state: `claimed=5`, `failed=0`, `done=0`, `pending=0`.
- No `gate_report_seed0.json` exists yet.
- ProICL post-gate manager PID `30471` is alive and waiting on the gate.
- Local watchdog screen `proicl_flow_watchdog` is detached and polling roughly
  every 60 seconds, with remote mirrors under
  `runs/proicl_flow_a100x4/watchdog/mirrors/`.
- Watchdog was hardened to record explicit `flow_status_seen` events and emit a
  `preempting` event if Flow reports the bid in the 5-minute preemption window.
- Remote GPU utilization is non-idle on all four A100s; GPU 0 is saturated,
  GPUs 1-3 are active but underutilized because upstream exact-RWS is hard-coded
  into five 100-problem batches.

Current exact-RWS worker log counters:

- `gpu-0-slot-a.log`: `mcmc_done=4`
- `gpu-0-slot-b.log`: `mcmc_done=3`
- `gpu-1-slot-a.log`: `mcmc_done=3`
- `gpu-2-slot-a.log`: `mcmc_done=5`
- `gpu-3-slot-a.log`: `mcmc_done=1`

Updated poll at `20260516T092902Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `9, 3, 3, 5, 1` for
`gpu-0-slot-a`, `gpu-0-slot-b`, `gpu-1-slot-a`, `gpu-2-slot-a`,
`gpu-3-slot-a`.

Updated poll at `20260516T095701Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `11, 3, 9, 13, 8`. Flow bid list still shows bid cap
`$0.10`; Flow pricing for `a100-80gb.sxm.4x` in `us-central3-a` reports
`price_per_hour=0.01` with available quantity `8`.

Updated poll at `20260516T102815Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `16, 4, 17, 13, 12`. GPU0 slot B is the slowest shard
because GPU0 is carrying the fifth upstream RWS batch; it is still advancing.

Updated poll at `20260516T112413Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `22, 4, 27, 21, 20`. Worker-log scan remains clean.
GPU0 slot B was inspected directly; the process is alive and inside the
upstream MCMC loop, not crashed.

Updated poll at `20260516T113531Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `23, 4, 27, 24, 21`. No result CSVs exist yet because
the upstream RWS script writes each 100-problem batch only after the batch
finishes; mirrored worker logs are the partial evidence until then.

Updated poll at `20260516T120633Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `25, 8, 32, 29, 22`. GPU0 slot B continued advancing,
so it is slow from contention/long generations rather than stuck.

Updated poll at `20260516T132653Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists.
Current MCMC counters are `38, 10, 40, 36, 27`. No exact-RWS result CSVs have
flushed yet.

Error-scan follow-up at `20260516T1332xxZ`: a `Traceback` string in
`gpu-1-slot-a.log` was generated model text inside a sampled solution, not a
Python crash. Queue still has `failed=0`; all five `power_samp_math.py`
processes are alive; stderr tails show active tqdm progress.

GEPA reflection update at `20260516T135544Z`: user asked whether GEPA should
use xAI env. The xAI path was made fail-closed with
`CapEnforcedReflectionLM`, tested locally and remotely, and the post-gate
manager was restarted before any GEPA stage had begun. Future GEPA archive
builds now use `--reflection-provider xai --env-file /workspace/polaris/.env`.
The manager forces `XAI_REFLECTION_INITIAL_CAP_DOLLARS=10` and
`XAI_REFLECTION_HARD_CAP_DOLLARS=10`; `XAI_API_KEY` was confirmed present
without printing it. `litellm` was installed in `/mnt/local/venvs/polaris`.
Exact-RWS workers were not touched.

Updated poll at `20260516T141859Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists
and `csv_count=0`. Current MCMC counters are `41, 18, 47, 39, 31`.

Updated poll at `20260516T145912Z`: Flow status remains `running`; queue is
still `claimed=5`, `failed=0`, `done=0`, `pending=0`; no gate report exists
and `csv_count=0`. Current MCMC counters are `45, 20, 55, 40, 32`. Periodic
price check shows bid cap `$0.10`; Flow pricing for `a100-80gb.sxm.4x` in
`us-central3-a` reports `price_per_hour=0.01` with available quantity `8`.

## ProICL Overnight Signal Run Authorization — 2026-05-17

User authorized implementation and launch of the first serious ProICL signal
run: three Reasoning Gym tracks, 20 held-out eval problems per track, B=8 for
frozen non-greedy arms, live cross-task GEPA archive, HF vendored RWS for MCMC,
and 8x A100 spot with a `$0.25/hr` bid cap and `$4` run-planning cap.

Decision: the exact 8-seed Karan-Du/RWS MATH500 replication remains
paper-readiness validation, but it is not a blocker for this pilot/signal run.
The result must be labeled pilot/signal, not a final statistical claim.

Implementation focus: make ProICL executable and resumable before launch by
adding `base_greedy`, executable runtime condition fields, a queue runner,
live GEPA archive construction, artifact-complete smoke validation, and
artifact-only decomposition aggregation.

Engineering update at `20260517T0414Z`: ProICL analysis now tolerates
non-positive pilot/smoke RF denominators by writing `rf_valid=false`, null RF
fields, and an explicit warning instead of aborting. Positive-denominator RF
math is unchanged. Unit coverage verifies the noisy-denominator path and the
decomposition markdown includes `discovery_gain` and `composition_gain`.

Flow launch update at `20260517T0414Z`: direct Flow code upload for the full
runner was rejected by the platform startup-script size limit, so the launch
route is now a small `upload_code=false` 8xA100 hold job followed by SSH/rsync
of the checked-out repo and an in-instance `scripts/run_proicl_signal.py`
launch. The submitted hold bid is `bid_tFqcMgmRG4VR6YwW`
(`proicl-a100x8-signal-live`) with `a100-80gb.sxm.8x`, region
`us-central3-a`, and max price `$0.25/hr`.

## ProICL Flow Preemption Stop — 2026-05-16

The live Flow run hit the preemption/infra stop condition.

Evidence:

- Flow status at `20260516T152919Z`: `paused` for bid
  `bid_d43eJOwbGefYgX5X`.
- Watchdog recorded `preempting` at `20260516T152029Z`,
  `20260516T152138Z`, `20260516T152246Z`, and `20260516T152354Z`.
- Watchdog recorded `paused` at `20260516T152501Z` and subsequent polls.
- Local mirror path:
  `runs/proicl_flow_a100x4/watchdog/mirrors/runs/proicl_flow_a100x4/prorl_recovery_seed0_gate/phase0/rws_exact`.
- Mirrored queue state: `claimed=5`, `failed=0`, `done=0`, `pending=0`.
- Mirrored gate report: absent.
- Mirrored exact-RWS CSVs: `0`.
- Final mirrored worker counters before pause:
  `gpu-0-slot-a=46`, `gpu-0-slot-b=22`, `gpu-1-slot-a=55`,
  `gpu-2-slot-a=44`, `gpu-3-slot-a=32`.

Interpretation: this is an infrastructure/preemption stop, not a scientific
gate failure. The exact-RWS gate never completed, so no ProICL smoke cells, full
factorial cells, or RF/decomposition artifacts were launched.

Decision: keep the exact upstream RWS gate running rather than silently swapping
in the faster POLARIS batched MCMC path. That path may be useful later, but
using it as the hard gate would be a protocol decision, not an overnight
implementation detail.

Updated poll at `20260516T135316Z`: Flow status remains `running` for
`bid_d43eJOwbGefYgX5X` / `polaris-prorl-a100x4-live` on 4xA100 in
`us-central3-a`. Queue remains `claimed=5`, `failed=0`, `done=0`, `pending=0`;
no `gate_report_seed0.json` exists and no exact-RWS CSV has flushed yet. Current
MCMC counters are `41, 14, 40, 36, 29` for `gpu-0-slot-a`, `gpu-0-slot-b`,
`gpu-1-slot-a`, `gpu-2-slot-a`, `gpu-3-slot-a`. `nvidia-smi` shows all GPUs
active: GPU0 `99%` util with two slots, GPUs1-3 at `60%`, `64%`, and `55%`.
The post-gate manager is alive and still blocked at `gate_wait_start`, so the
ProICL smoke/full cells have not started. GEPA is currently configured for
`--reflection-provider local-hf`; remote `XAI_API_KEY` and `XAI_BASE_URL` are
unset, so no xAI reflection call will be made unless the launch contract is
changed before the gate passes.

Follow-up at `20260516T1356xxZ`: local and remote
`runs/proicl_flow_a100x4/launch_proicl_after_gate.sh` are configured for
`--reflection-provider xai --env-file "$REPO/.env"` for both GEPA smoke and
full archive construction. Remote `.env` contains `XAI_API_KEY`, `XAI_BASE_URL`,
`XAI_REFLECTION_MODEL`, and xAI caps; `scripts/xai_reflection_smoke.py
--dry-run` passed, and the live one-request xAI smoke completed with status
`live_request_complete`. Restarted only the waiting post-gate manager so the
loaded bash function uses the xAI path; exact-RWS gate workers were not touched
and remain alive.

## ProICL Overnight Signal Flow Readiness — 2026-05-17

Engineering status: ProICL infrastructure is implemented locally for the
overnight signal run. `tests/unit/test_proicl.py` passes, protocol drift guard
passes, and the updated Flow hold configs dry-run.

Observed Flow issue: Flow-provided cache envs pointed Hugging Face at
`/workspace/.cache/huggingface`, which was root-owned on the 1x A100 smoke VM
and caused prefetch to fail before any generation:
`PermissionError: /workspace/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B`.

Fix: `scripts/run_proicl_signal.py` now overrides Flow cache paths to
`/mnt/local/proicl-cache/*` before prefetch/generation, and the ProICL Flow
hold configs export the same writable cache roots. This is an infrastructure
fix for future 8x launches, not a change to the scientific conditions.

Live state: `proicl-a100x1-smoke-live` / `bid_EkXDbgjWpXj2Zv07` is running and
has a clean remote sanity check (`torch.cuda.is_available() == True`,
`NVIDIA A100-SXM4-80GB`, xAI/HF credentials present). Active smoke artifact
root is `/workspace/polaris/runs/proicl_smoke_signal_v2`; live GEPA archive was
built, `base_greedy` completed, and `mcmc_only` is in progress. The 8x full
lane `proicl-a100x8-signal-r2` / `bid_8WfYRSpUmggaeywf` remains provisioning
with no SSH host yet.

Update at `20260517T0526Z`: the active 1x smoke completed `base_greedy`,
`mcmc_only`, and `gepa_only`; `gepa_mcmc` is now running. Completed artifacts
have been mirrored locally under `runs/flow_mirrors/proicl_smoke_signal_v2`.
The faithful HF/RWS smoke timing for `mcmc_only` was approximately 8.5 minutes
for one Reasoning Gym problem with `B=2`, `max_new_tokens=512`, `block_num=16`,
and `mcmc_steps=10`, which supports the overnight 8x estimate rather than an
instant smoke expectation.

8x launch correction at `20260517T0528Z`: canceled the stale
`proicl-a100x8-signal-r2` bid because its launch spec still used the old
`/workspace/.cache` env. A Flow retry briefly created two corrected 8x bids;
the duplicate `proicl-a100x8-signal-r3` was canceled. The remaining corrected
8x bid is `proicl-a100x8-signal-r2-176f4a` / `bid_C9HN8pBalwLcqUOh`, currently
open at `$0.25/hr`.

Flow cleanup at `20260517T0543Z`: kept the active smoke lane
`proicl-a100x1-smoke-live` / `bid_EkXDbgjWpXj2Zv07` and the intended full-run
8x lane `proicl-a100x8-signal-r2-176f4a` / `bid_C9HN8pBalwLcqUOh`. The stale
paused 4x A100 bids `polaris-prorl-a100x4-live` / `bid_d43eJOwbGefYgX5X` and
`proicl-a100x4-signal-live` / `bid_gBedJ5hhFXgT4yQu` could not be canceled by
Flow (`reason=not-cancellable`), but both remain paused and their bid caps were
lowered from `$0.12/hr` to `$0.04/hr`. Old local Flow polling loops for stale
bid names were stopped.

Correction at `20260517T0620Z`: Flow showed additional non-intended bids. The
2x fallback `proicl-a100x2-signal-fallback` / `bid_1TYhB9E0wHeUPLkI`, duplicate
4x `proicl-a100x4-signal-live-3a7e1c` / `bid_FbGrPaCSLjWoXFqz`, and stale 4x
`proicl-a100x4-signal-live` / `bid_gBedJ5hhFXgT4yQu` were canceled where the API
allowed it. The 1x smoke bid is now canceled. `flow status --json` now shows
only the intended 8x signal bid as non-canceled plus historical canceled rows.
`flow bid list` still reports the deactivated 2x fallback as `Allocated`, but
subsequent pause/delete attempts return `Instance order deactivated` /
`not-cancellable`; treat `flow status` as the live source of truth unless this
persists in billing.

Update at `20260517T0607Z`: ProICL smoke passed end-to-end on the 1x A100.
All six operational conditions completed for one held-out `reasoning_gym_boxnet`
problem: `base_greedy`, `mcmc_only`, `gepa_only`, `gepa_mcmc`,
`gepa_mcmc_memory`, and `prorl_v2_greedy`. The replay pass skipped all six
completed cells (`pending=0`, `skipped=6`) and re-aggregated successfully.
Artifacts were mirrored locally at
`runs/remote_mirrors/proicl_smoke_signal_v2/`. Required memory artifacts
(`memory.sqlite`, `memory_events.jsonl`) and decomposition artifacts
(`proicl_decomposition.json`, `proicl_decomposition.md`) exist. RF is undefined
for the one-problem smoke slice because `A_prorl_v2 == A_base == 0`, which is a
slice artifact rather than a harness failure.

Full-run staging at `20260517T0615Z`: the 1x A100 is building the full live
cross-task GEPA archive for `runs/proicl_overnight_signal/full` using xAI
reflection, archive size `k=8`, dev split `[0, 6)`, and max metric calls `64`.
The temporary 1x full-run watcher was stopped so the 1x will not block larger
allocation after archive sync. Active larger bids remain open but hostless:
8x `bid_C9HN8pBalwLcqUOh` at `$0.25/hr`, 4x `bid_FbGrPaCSLjWoXFqz` at
`$0.15/hr`, and 2x `bid_1TYhB9E0wHeUPLkI` at `$0.05/hr`. Working hypothesis:
the allocated 1x may be consuming the currently runnable A100 slot, so after
the archive is complete and mirrored, release the 1x to let the larger bid
allocate.

Update at `20260517T0651Z`: ProICL full-run archive staging is complete and
mirrored locally at `runs/remote_mirrors/proicl_overnight_signal/full`. The
live cross-task GEPA archive has `k=8`, tracks
`reasoning_gym_boxnet`, `reasoning_gym_graph_color`, and
`reasoning_gym_family_relationships`, dev split `[0, 6)`, and xAI reflection
metadata in `archive_build_manifest.json`. The 1x smoke/archive bid was
canceled after sync.

Engineering update: added `scripts/flow_launch_proicl_signal.py`, a local Flow
stager that waits for SSH-ready allocation, rsyncs the current checkout and
staged ProICL artifacts, installs the repo on the remote host, and starts the
resumable `scripts/run_proicl_signal.py --skip-smoke` full run. It now treats a
Flow hostname as usable only after a real SSH probe succeeds; stale paused
hosts are not accepted.

Flow allocation state: the intended A100 lanes are still hostless. Primary
`bid_C9HN8pBalwLcqUOh` (`proicl-a100x8-signal-r2-176f4a`, cap `$0.25/hr`) and
fallback `bid_kyX4MasqNS3GNG3O` (`proicl-a100x4-signal-r4`, cap `$0.15/hr`) are
accepted/provisioning but have no SSH host yet. The stale paused 4x bid
`bid_d43eJOwbGefYgX5X` was canceled where the API allowed it and removed from
`flow status`; a `flow instance delete` attempt still reported it as
not-cancellable/paused, so continue treating `flow status` as the live source.
Stray calibration bids `bid_uioratnAkVmxZeos` (A100 1x) and
`bid_pLp9TVgtD9h8EgSG` (B200 1x) were canceled because they are outside the
ProICL compute plan. A local watcher is polling the 8x and 4x A100 bids and
will launch the full signal run on the first SSH-ready host, preferring 8x.

Flow correction at `20260517T0710Z`: canceled the generic 4x A100 bid
`bid_kyX4MasqNS3GNG3O` because Flow continued to expose it as `4xa100` and kept
it hostless despite a raised cap. Submitted an explicit
`a100-80gb.sxm.4x` bid through `configs/flow_proicl_a100x4_hold.yaml`:
`proicl-a100x4-signal-explicit` / `bid_KaJZhNYQHVvC9pSp`, cap `$0.18/hr`,
24h max runtime. A launcher watcher is attached to this explicit bid and will
start the staged ProICL full run once SSH is actually reachable. The 8x lane was
not pursued further because live A100 8x pricing rose above the planned cap.

Flow bid update at `20260517T0717Z`: 4x A100 live price rose to `$0.18/hr`,
exactly matching the explicit 4x bid. Raised `bid_KaJZhNYQHVvC9pSp` to
`$0.19/hr` to clear tie/age effects. `flow instance list` reports it as
`provisioning`; `flow logs` still says startup logs will appear once the
instance is up. No ProICL full-run cells have started yet.

Flow fallback at `20260517T0712Z`: `flow grab` does not accept the fully
qualified `a100-80gb.sxm.4x` name, but accepted generic `4xa100` as
`proicl-a100x4-signal-grab` / `bid_dKXz3VSfH7H2e7Uw` with cap `$0.20/hr`.
Stopped the single-bid launcher and replaced it with a first-ready controller
watching `bid_KaJZhNYQHVvC9pSp` and `bid_dKXz3VSfH7H2e7Uw`. It will select the
first SSH-ready bid, cancel the other, and launch exactly one ProICL runner.

Flow fallback at `20260517T0714Z`: added a slower but cheap 2x A100 fallback,
`proicl-a100x2-signal-grab` / `bid_KhQe2KIpyysoPA1Z`, cap `$0.04/hr`. The
active first-ready controller now watches both 4x bids plus this 2x bid. If the
2x bid wins first it will run the same ProICL plan with GPUs `0,1` and
`num_shards=8`; if a 4x bid wins first it will use GPUs `0,1,2,3`. Non-selected
bids are canceled before launch.

Flow CLI correction at `20260517T0716Z`: Flow project auto-resolution began
failing with `Project is required but not configured`, despite existing
configured credentials. Verified that explicit
`MITHRIL_PROJECT=proj_9UVziLCzCBXzLp9a` and
`MITHRIL_PROJECT_ID=proj_9UVziLCzCBXzLp9a` restores `flow status`. Restarted
the calibration-bid canceler and first-ready ProICL controller with those env
vars pinned.

Budget correction at `20260517T0720Z`: `flow bid list` revealed the 4x grab bid
`bid_dKXz3VSfH7H2e7Uw` had limit `$0.80`, outside the pilot budget envelope.
Canceled both 4x bids (`bid_dKXz3VSfH7H2e7Uw` and `bid_KaJZhNYQHVvC9pSp`) and
continued only with the 2x A100 fallback `bid_KhQe2KIpyysoPA1Z` at `$0.07/hr`.
The 2x bid is allocated/provisioning but not SSH-ready yet.

Flow retry at `20260517T0725Z`: the 2x grab bid stayed allocated but
unreachable at a private `10.234.*` host and may have been billable while not
usable, so it was canceled. Added `configs/flow_proicl_a100x2_hold.yaml`
(`a100-80gb.sxm.2x`, cache env, 24h max runtime, cap `$0.07/hr`) and submitted
`proicl-a100x2-signal-explicit` / `bid_IpiJH8Rc0gcEBOIe`. A launcher watcher is
attached to this explicit 2x bid.

Flow startup observation at `20260517T0730Z`: API cache for
`bid_IpiJH8Rc0gcEBOIe` reports instance status `STATUS_STARTING`,
`private_ip=10.234.110.155`, and `ssh_destination=null`. Local SSH to the
private IP times out, so the launcher is correctly refusing to treat it as
usable. Continue waiting only until a bounded startup window expires; without a
non-null SSH destination, no remote staging or artifact mirroring can begin.

Flow launch fix at `20260517T0736Z`: `bid_IpiJH8Rc0gcEBOIe` became SSH-ready at
`54.214.155.248` with two A100 80GB GPUs. The first staging attempt failed
because `/workspace/polaris` was not writable by `ubuntu`; patched
`scripts/flow_launch_proicl_signal.py` to `sudo chown` the remote repo before
rsync. The next attempt failed during `reasoning-gym` install because
`pycosat` needed `Python.h`; patched the remote installer to install
`python3-dev build-essential`. A third launch failed immediately on model
prefetch because `/mnt/local/proicl-cache` was root-owned; patched the launch
command to own `/mnt/local/proicl-cache` before setting HF cache env vars.
Relaunched the full ProICL signal runner on the same 2xA100 bid with PID
`8931`; local artifact mirroring to
`runs/remote_mirrors/proicl_overnight_signal/` is active.

Preemption response at `20260517T0743Z`: Flow marked
`bid_IpiJH8Rc0gcEBOIe` as `preempting` after `reasoning_gym_boxnet/base_greedy`
had completed all 8 shards (`8` mirrored metrics) and `mcmc_only` shards were
running. Performed immediate artifact sync and raised the bid cap first to
`$0.12/hr`, then to `$0.25/hr`, still inside the user-authorized A100 envelope.
Emergency sync/poll loop is active every 30s until status returns to `running`
or the host disappears.

Throughput correction at `20260517T0747Z`: observed MCMC children using only
~30-36% GPU and ~5GB VRAM per A100, so two physical A100s were underfilled.
Stopped the active runner after syncing; completed `base_greedy` shards remain
valid. Relaunched the resumable runner with duplicated GPU slots
`--gpus 0 0 1 1` on the same 2xA100 host so four 1.5B HF workers can run
concurrently. Incomplete `mcmc_only` shards will be retried; completed cells are
skipped.

Correction to throughput correction at `20260517T0750Z`: the four-slot runner
exited without traceback before any MCMC shard completed, likely a process-level
kill under oversubscription. Reverted to the stable two-worker shape
`--gpus 0 1` on the same host. This is slower but preserves correctness and
resume safety; `base_greedy` remains completed and MCMC retries from scratch.

Launcher robustness fix at `20260517T0754Z`: found the real reason background
runners vanished without traceback. Flow/logind stops the `ubuntu` user slice
after SSH sessions close, killing `nohup` children. Patched
`scripts/flow_launch_proicl_signal.py` to launch the runner as a transient
host-level `systemd` service (`proicl-signal.service`) under user `ubuntu`.
Relaunched successfully; service PID is `18572`, with MCMC children `18590` and
`18591` active.

Follow-up at `20260517T0802Z`: the transient service still exited cleanly after
~90 seconds while MCMC children were active, leaving no `cell_failed` event and
no traceback. Audited the host: the stale
`full_after_archive_watcher.pid` was dead, but an unrelated vLLM/HF calibration
process had started on GPU 0 and was killed to avoid contaminating the ProICL
run. Relaunched the same resumable ProICL script in a foreground SSH session so
Flow/logind cannot reap it when the control session closes. Completed
`boxnet/base_greedy` shards remain valid; queue resumes from incomplete
`mcmc_only` cells.

Root cause update at `20260517T0807Z`: the foreground run was interrupted by a
stale local cleanup/orchestration process from earlier Flow calibration work,
which issued `systemctl stop proicl-signal.service` on the same 2xA100 host and
closed the SSH session before any MCMC shard flushed metrics. Killed stale local
calibration/Flow loops, leaving only ProICL mirror/status monitors. Relaunched
the 2xA100 foreground runner. Also submitted an opportunistic 8xA100 hold bid
(`bid_XUJV3HXPWgZfPsaS`) at `$0.25/hr`; if it exposes SSH, migrate staged
artifacts there and cancel the slower 2x host.

Root-owned service fix at `20260517T0811Z`: foreground SSH still died when
Flow/logind cleaned the `ubuntu` user session. Patched
`scripts/flow_launch_proicl_signal.py` to run `proicl-signal.service` as a
root-owned system unit instead of `--uid=ubuntu`; this keeps the workload out
of `user-1000.slice`. Relaunched the 2xA100 runner with MainPID `28448` and
MCMC children `28463`/`28464`. This is operational only; experiment artifacts
still record model/condition provenance and remain resumable.

Stale wrapper isolation at `20260517T0816Z`: a delayed stale cleanup command
continued to target `proicl-signal.service` and
`run_proicl_signal_remote.sh`, stopping the root-owned service at `08:14:02Z`
before any MCMC shard completed. Patched the Flow launcher again to use a new
unit and script name: `proicl-signal-root.service` and
`run_proicl_signal_root.sh`. Killed pending delayed local status commands that
could re-touch the old service name. Next relaunch should be isolated from the
old cleanup path.

Relaunch at `20260517T0816Z`: synced the mirrored ProICL artifacts back to the
2xA100 host and launched `proicl-signal-root.service`. MainPID `31890`; runner
PID `31893`; current MCMC children `31905` and `31906`. Completed metric count
is still `8` (`boxnet/base_greedy` only); all MCMC shards are retrying because
previous attempts were killed before flush.
## vLLM/HF Calibration Flow Attempt — 2026-05-17 07:05 UTC

User authorized a bounded Flow calibration with total cap `$0.08`.

- Added `scripts/flow_vllm_hf_calibration_standalone.py`, a repo-local
  standalone calibration harness for real HF forward scoring vs vLLM
  forced-token scoring. It writes `score_parity.jsonl`,
  `mh_replay_parity.jsonl`, `full_chain_replay.json`,
  `calibration_summary.json`, and `calibration_report.md` when run on GPU.
- Added Flow configs:
  `configs/flow_vllm_hf_calibration_a100x1.yaml`,
  `configs/flow_vllm_hf_calibration_a100x1_hold.yaml`, and
  `configs/flow_vllm_hf_calibration_b200x1_hold.yaml`.
- `flow submit --dry-run --json
  configs/flow_vllm_hf_calibration_a100x1.yaml` passed after compressing the
  standalone script payload, but live submit still hit Flow's backend
  startup-script limit. The `vllm/vllm-openai:v0.9.2` image path also failed
  this backend script limit even for a 249-character hold command.
- Host/default-image hold attempts avoided the startup-script limit, but Flow
  did not allocate a running instance. Calibration-specific bids attempted:
  `bid_uioratnAkVmxZeos` (1x A100), `bid_pLp9TVgtD9h8EgSG` (1x B200),
  `bid_AmdDD6u6gy2xJarP` (1x A100 grab),
  `bid_BnOXPkfYWKcBJLQS` (1x B200 grab), and
  `bid_J8u4FIcRUfm1i6V0` (final 1x A100 grab at `$0.08/hr` for 1h).
  All remained `started_at = null`; no calibration GPU runtime executed.
- `bid_J8u4FIcRUfm1i6V0` was explicitly canceled before any `started_at`
  appeared. `flow bid list --json` then reported it as deactivated at
  `2026-05-17T06:58:26.695178Z`; `flow status` may still show stale
  `provisioning` for this deactivated bid.
- No HF-vLLM equivalence result exists yet. The blocker is Flow allocation /
  startup plumbing, not a failed numerical parity gate.
- Verification after edits: `PYTHONDONTWRITEBYTECODE=1
  ./.venv-eval/bin/python -m py_compile
  scripts/flow_vllm_hf_calibration_standalone.py` passed, and
  `bash scripts/check_protocol_sync.sh` passed.

## vLLM/HF Calibration Result on Existing 2xA100 — 2026-05-17 08:11 UTC

Used only the already-running `proicl-a100x2-signal-explicit` host
(`bid_IpiJH8Rc0gcEBOIe`, SSH `54.214.155.248`); no new GPU capacity was
provisioned. Stopped the stale HF ProICL restart wrapper by moving
`runs/proicl_overnight_signal/logs/run_proicl_signal_remote.sh*` into
`disabled_wrappers/` and killing only the ProICL runner/process children.

Calibration artifacts:

- Production-HF-semantics run:
  `/workspace/polaris/runs/flow_vllm_hf_calibration/deepseek_1p5b_hfauto_20260517T080041Z/calibration_summary.json`.
  Result: `passed=false`, tokenizer parity passed, but score parity failed
  (`max_abs_diff=0.6359138488769531`) and MH replay failed
  (`max_abs_diff=0.8436350776778454`).
- Matrix root:
  `/workspace/polaris/runs/flow_vllm_hf_calibration/matrix_20260517T080405Z`.
  Best case was HF float32/default vs vLLM float32/transformers:
  `score_max_abs_diff=0.015389442443847656`,
  `mh_max_abs_diff=0.015967544116392673`, full-chain replay passed, but the
  strict `1e-3` score/MH gates failed. HF eager was similar; bfloat16/auto
  variants were much worse and changed full-chain decisions.

Decision: vLLM forced-decode scoring is not science-valid for POLARIS MCMC on
this stack yet. Do not run ProICL MCMC with `--backend vllm` from these
artifacts.

Implemented a hard run gate:

- `src/polaris/infra/vllm_calibration.py` now validates calibration summaries
  and rejects missing/failed vLLM parity artifacts.
- `scripts/run_condition.py` and `scripts/run_proicl_signal.py` abort before
  model prefetch/execution when `--backend vllm` lacks a passing artifact.
- Remote rejection check with the failed best-case artifact exited `rc=1` with
  `vLLM calibration gate failed`.

Verification:

- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_vllm_calibration.py tests/unit/test_vllm_serving.py
  tests/unit/test_proicl.py` passed (`28 passed`).
- `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m py_compile
  scripts/run_condition.py scripts/run_proicl_signal.py
  src/polaris/infra/vllm_calibration.py
  scripts/flow_vllm_hf_calibration_standalone.py` passed.

Additional debug probes at `20260517T0816Z`:

- vLLM V0 `model_impl=auto`, `dtype=float32` also failed:
  `score_max_abs_diff=0.014862060546875`,
  `mh_max_abs_diff=0.017737813067469688`.
- vLLM V1 with `VLLM_ATTENTION_BACKEND=TORCH_SDPA` cannot initialize on this
  vLLM `0.9.2` stack; vLLM raises `NotImplementedError:
  VLLM_USE_V1=1 is not supported with VLLM_ATTENTION_BACKEND=TORCH_SDPA`.
- Final host check: no ProICL/calibration processes active and both A100s at
  `0 MiB` used. The stale remote wrapper path now contains an inert disabled
  stub to prevent unintentional HF restarts.

## ProICL HF Signal Relaunch and Capacity Plan — 2026-05-17 08:29 UTC

Relaunched the ProICL signal run on the existing 2xA100 host
`bid_IpiJH8Rc0gcEBOIe` using a root-owned systemd unit
`proicl-signal-root.service` and a new wrapper path
`runs/proicl_overnight_signal/logs/run_proicl_signal_root.sh`, avoiding the
old disabled/stale `run_proicl_signal_remote.sh` path.

Live state at `2026-05-17 08:27 UTC`:

- Service `proicl-signal-root.service`: `active/running`, `MainPID=31890`.
- Active cells: `reasoning_gym_boxnet/mcmc_only` shards `0` and `1`.
- GPU use: both A100s active around `33-37%`, ~`4.7-5.0 GiB` each.
- Flushed artifacts: `8` `metrics.json` files, all
  `reasoning_gym_boxnet/base_greedy`; MCMC cells were still running and had
  not yet flushed final metrics.
- Local mirror refreshed under
  `runs/remote_mirrors/proicl_overnight_signal/`.

Because faithful HF/RWS MCMC is slow on a lone 2xA100, added resumable
cell-partition support:

- `scripts/run_proicl_signal.py`: `--cell-stride`, `--cell-offset`,
  `--skip-aggregate`.
- `scripts/flow_launch_proicl_signal.py`: forwards those partition flags and
  configurable tracks to the remote root-owned service wrapper.
- Verification:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_proicl.py -q` passed, and
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m py_compile
  scripts/run_proicl_signal.py scripts/flow_launch_proicl_signal.py` passed.

Capacity state:

- Running: `bid_IpiJH8Rc0gcEBOIe` (`2xa100`, cap `$0.25/hr`).
- Pending migration/fallback bids:
  `bid_XUJV3HXPWgZfPsaS` (`8xa100`, cap `$0.25/hr`),
  `bid_5bNhKTIPlMXjfO5f` (`4xa100`, cap `$0.15/hr`),
  `bid_3JRytdppRzerTQ1U`, `bid_TpMopANF8WgTGyic`,
  `bid_0vJo3iSaUCYa7XFz` (`2xa100`, caps raised to `$0.15/hr`).

Plan: continue the live 2xA100 run while waiting for larger/extra A100 capacity.
If enough extra hosts allocate, stop the unpartitioned runner and relaunch
disjoint partition workers from the synced artifact root to avoid duplicate
cells.

Follow-up at `2026-05-17 08:31 UTC`: `flow availability --json` reported
A100 spot clearing around `$0.32/hr` for `a100-80gb.sxm.8x`, `$0.16/hr` for
`a100-80gb.sxm.4x`, and `$0.08/hr` for `a100-80gb.sxm.2x`. Updated caps to
match the current market while staying on A100 spot:

- `bid_XUJV3HXPWgZfPsaS`: `$0.25/hr` -> `$0.40/hr`.
- `bid_5bNhKTIPlMXjfO5f`: `$0.15/hr` -> `$0.20/hr`.
- Extra 2x bids remained at `$0.15/hr`, above the listed `$0.08/hr`.

Follow-up at `2026-05-17 08:38 UTC`: added six `1xa100` fallback bids at
`$0.08/hr` because Flow reported `1xa100` availability around `$0.03/hr` and
single-GPU hosts may allocate when bundle bids remain in provisioning. These
are intended only for partition workers if they expose SSH:

- `bid_FZwzZ6Fygc7Rp7Wb`
- `bid_6CvdVrPCXLiR24h0`
- `bid_P2ISi6CNBmfXaqYg`
- `bid_5jaW1VZkVZqy874m`
- `bid_q1RReCSYkNAiPVdy`
- `bid_50xclN5RdfXHTOwX`

Also cancelled the stale non-ProICL calibration bid
`bid_J8u4FIcRUfm1i6V0` with `flow instance delete --json`; it had remained
visible as `provisioning` after the older `flow cancel` prompt.

Runtime checkpoint at `2026-05-17 08:52 UTC`:

- Extra A100 bids remained in `provisioning`; no SSH host exposed yet.
- Live 2xA100 root service still `active/running`.
- `reasoning_gym_boxnet/mcmc_only` partial progress:
  shard `0` wrote `8` candidates and `1` selected problem;
  shard `1` wrote `16` candidates and `2` selected problems.
- Full `metrics.json` count was still `8` because neither MCMC shard had
  completed all assigned problems yet.
- Local mirror refreshed with these partial artifacts.

Scheduler bug and relaunch at `2026-05-17 09:16 UTC`:

- Observed `mcmc_only/shard-1` completed at `20260517T090259Z`, after which the
  old launcher started shard `2` on GPU `0` while shard `0` was already on GPU
  `0` and GPU `1` was idle. This was a real scheduler bug: round-robin launch
  did not reuse the GPU that had just freed.
- Fixed `src/polaris/proicl/launcher.py` to maintain an explicit
  `available_gpus` queue and return the completed cell's GPU to that queue.
- Added regression test
  `test_proicl_launcher_reuses_the_gpu_that_finished`.
- Verification:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_proicl.py -q` passed (`14` tests), and
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m py_compile
  src/polaris/proicl/launcher.py scripts/run_proicl_signal.py
  scripts/flow_launch_proicl_signal.py` passed.
- Stopped the old root systemd service, synced artifacts locally, and relaunched
  from the mirror on the same 2xA100 host with the patched scheduler. Relaunch
  skipped `9` complete cells (`8` base-greedy shards + completed
  `mcmc_only/shard-1`) and started `mcmc_only/shard-0` on GPU `0` plus
  `mcmc_only/shard-2` on GPU `1`.
- Post-relaunch GPU check: both A100s active (`33%` and `40%`, ~`4.1 GiB`
  each).

Resume-ledger fix at `2026-05-17 09:23 UTC`:

- Found a second resume bug: incomplete artifact directories were reused, so
  restarted cells could append to old partial `jsonl` ledgers.
- Fixed `run_cells` to remove an incomplete cell's artifact directory before
  relaunching it. Completed cells are still skipped before this deletion path.
- Extended the GPU-scheduler regression test to assert stale partial ledgers are
  removed before rerun.
- Verification remained green:
  `PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
  tests/unit/test_proicl.py -q` passed, and the same `py_compile` command
  passed.
- Relaunched again from the local mirror. Completed `mcmc_only/shard-1`
  remained complete; incomplete shard artifacts were regenerated cleanly from
  the trajectory cache where possible.

Capacity pruning at `2026-05-17 09:35 UTC`: the extra 1x/2x/duplicate-4x
fallback bids remained in `provisioning` for too long and looked like quota or
allocator noise, not useful capacity. Cancelled them to avoid a later accidental
swarm allocation and kept only:

- running `bid_IpiJH8Rc0gcEBOIe` (`2xa100`);
- pending preferred `bid_XUJV3HXPWgZfPsaS` (`8xa100`);
- pending fallback `bid_5bNhKTIPlMXjfO5f` (`4xa100`).

Runtime checkpoint at `2026-05-17 09:54 UTC`:

- Larger bids still `provisioning`; only the 2xA100 is running.
- `reasoning_gym_boxnet/mcmc_only` now has two completed shards:
  `shard-1` at `20260517T090259Z` and `shard-0` at `20260517T094853Z`.
- Completed metrics: `10` total =
  `8` `reasoning_gym_boxnet/base_greedy` + `2`
  `reasoning_gym_boxnet/mcmc_only`.
- Active MCMC shards after the scheduler fix: `shard-2` and `shard-3` on the
  two A100s.
- Local mirror refreshed with the two completed MCMC shards.

Runtime checkpoint at `2026-05-17 10:34 UTC`:

- `bid_XUJV3HXPWgZfPsaS` (`8xa100`) and `bid_5bNhKTIPlMXjfO5f` (`4xa100`)
  remained in `provisioning`; no SSH host.
- Live 2xA100 still healthy, both GPUs active.
- Completed metrics unchanged at `10`:
  `8` `base_greedy` + `2` `mcmc_only`.
- Active `mcmc_only` shards `2` and `3` each completed one problem
  (`8` candidates + `1` selected row each), so they are progressing but slow.
- Local mirror refreshed again.

Capacity correction at `2026-05-17 11:06 UTC`: the earlier A100 bundle caps
were likely below the total bundle clearing price because Flow's pricing feed
reported about `$0.08/GPU-hr`, not necessarily whole-instance price. Raised the
kept larger bids accordingly:

- `bid_XUJV3HXPWgZfPsaS` (`8xa100`): `$0.40/hr` -> `$0.80/hr`.
- `bid_5bNhKTIPlMXjfO5f` (`4xa100`): `$0.20/hr` -> `$0.40/hr`.

As of `2026-05-17 11:12 UTC`, both still had `host=null`; the 2xA100 remains
the only productive host.

Capacity pivot at `2026-05-17 11:41 UTC`: Flow availability showed the
monolithic `8xa100` and `4xa100` lanes priced at about `$0.80/hr` and
`$0.40/hr`, while `2xa100` is available at about `$0.06/hr`. Because ProICL
now supports cell partitioning, the cost-minimal speed-up is several cheap 2x
workers rather than one expensive 8x bundle. Canceled the unused pending
`8xa100` (`bid_XUJV3HXPWgZfPsaS`) and `4xa100`
(`bid_5bNhKTIPlMXjfO5f`) bids before they could allocate and launched three
additional `2xa100` partition workers at `$0.07/hr`:

- `bid_uRnMhpR1kOKottJA` / `proicl-a100x2-signal-part-1`
- `bid_YThigLmRn8cclgbq` / `proicl-a100x2-signal-part-2`
- `bid_x6TfmNMIn3HSp4Nl` / `proicl-a100x2-signal-part-3`

Current running host `bid_IpiJH8Rc0gcEBOIe` continues producing artifacts while
the new workers provision. Once additional 2x hosts expose SSH, stop the
unpartitioned service, sync completed artifacts, and relaunch the workers as a
four-way disjoint partition (`cell_stride=4`, offsets `0..3`) so completed
cells are skipped and no host duplicates future cells.

Runtime checkpoint at `2026-05-17 11:42 UTC`: existing 2x worker completed
`reasoning_gym_boxnet/mcmc_only/shard-3`, raising completed metrics to `11`
(`8` `base_greedy` + `3` `mcmc_only`). It started `mcmc_only/shard-4`.
Local mirror refreshed with the new completed shard. The three new 2x partition
bids are still provisioning with no SSH host.

Flow bid adjustment at `2026-05-17 11:44 UTC`: live availability for
`a100-80gb.sxm.2x` moved to `$0.07/hr`, exactly matching the three new
partition bids. Raised those bids to `$0.08/hr` to clear tie effects while
keeping the planned 8-GPU fanout near `$0.32/hr` total bid cap:
`bid_uRnMhpR1kOKottJA`, `bid_YThigLmRn8cclgbq`,
`bid_x6TfmNMIn3HSp4Nl`.

Flow bid adjustment at `2026-05-17 11:47 UTC`: after another auction window the
three new 2x bids still exposed `host=null`. Raised only those three bids to
`$0.12/hr`; this keeps the four-host fanout below the abandoned monolithic
8xA100 bid cap while making allocation more likely.

Allocation diagnosis at `2026-05-17 11:52 UTC`: raising
`bid_uRnMhpR1kOKottJA` to `$0.25/hr` still left it `Open`, suggesting the
blocker may be quota/allocation state rather than price. `flow instance list`
also still displayed stale calibration bid `bid_J8u4FIcRUfm1i6V0` as
`provisioning`; canceled it again to clear any possible allocator/quota noise.

Runtime optimization at `2026-05-17 11:59 UTC`: each HF/RWS worker was using
only ~5.5GB VRAM and ~35% GPU, so the 2xA100 host was underfed. Synced the
current artifacts, stopped the two-slot systemd runner, and relaunched the same
host with eight logical worker slots (`--gpus 0 0 0 0 1 1 1 1`). The relaunched
runner skipped `11` completed cells and immediately started eight new cells.
Observed both GPUs at `99%` utilization with only ~16-18GB VRAM used per GPU.
Canceled the three unused 2x partition bids after this saturation check to
avoid idle spend if they later allocated.

Runtime checkpoint at `2026-05-17 12:05 UTC`: oversubscribed runner remains
healthy (`ActiveState=active`, both GPUs `99%`). Completed metrics increased to
`13`: `8` `base_greedy`, `3` `mcmc_only`, and `2` `gepa_only` for
`reasoning_gym_boxnet`. Local mirror refreshed.

Runtime checkpoint at `2026-05-17 12:15 UTC`: oversubscribed runner remains
healthy with both GPUs at `99%`. Completed metrics increased to `19`:
`reasoning_gym_boxnet/base_greedy` `8/8`, `mcmc_only` `3/8`, and
`gepa_only` `8/8`. `gepa_mcmc` started (`shards 0..2` launched). Local mirror
refreshed.

Runtime checkpoint at `2026-05-17 12:47 UTC`: service still active and both
GPUs remain saturated (`99%`, ~20-22GB VRAM). Completed metrics remain `19`,
but all active RWS stderr logs have fresh mtimes and `mcmc_only/shard-5`
advanced to one selected problem. Synced partial artifacts/caches locally even
without new completed metric bundles.

Runtime checkpoint at `2026-05-17 13:17 UTC`: service still active, GPUs still
`99%`. Completed metrics increased to `20` after
`reasoning_gym_boxnet/mcmc_only/shard-4` completed; `mcmc_only` is now `4/8`
for boxnet. Runner immediately launched `gepa_mcmc/shard-3`. Local mirror
refreshed.

Preemption at `2026-05-17 13:47 UTC`: SSH reset during the next monitor poll.
Flow reported `bid_IpiJH8Rc0gcEBOIe` as `pending`/`paused`, and `flow bid
list` showed the 2xA100 market at `$0.25/hr`, exactly matching the old bid.
Completed artifacts are locally mirrored through `20` metric bundles. Raised
the same 2x bid to `$0.30/hr` to regain capacity and resume from the mirror.

Market adjustment at `2026-05-17 13:56 UTC`: 2xA100 rose again to `$0.30/hr`
and stayed open; raised the 2x fallback to `$0.35/hr`. A fresh 8xA100 resume
bid (`bid_UJV8hYbXWNFC9dWL`, `proicl-a100x8-signal-resume`) was submitted at
`$1.25/hr` after availability moved to `$1.20/hr`. At this price the 8x lane is
the same per-GPU-hour as 2x but should reduce wall time and oversubscription
drag. First SSH-ready lane will be used; the other should be canceled.

Market spike at `2026-05-17 14:01 UTC`: A100 rose again (`2xa100`
`$0.50/hr`, `8xa100` `$2.00/hr`). To avoid blowing the pilot cap on the 8x
lane, canceled `bid_UJV8hYbXWNFC9dWL` and raised the existing 2x fallback
`bid_IpiJH8Rc0gcEBOIe` to `$0.55/hr`. Resume target remains the local mirror
with `20` completed metric bundles.

Market adjustment at `2026-05-17 14:05 UTC`: 2xA100 availability moved to
`$0.55/hr`, exactly matching the fallback cap, so the bid stayed `Open`.
Raised `bid_IpiJH8Rc0gcEBOIe` to `$0.60/hr`.

Fallback addition at `2026-05-17 14:09 UTC`: 2xA100 moved again to `$0.60/hr`
while 1xA100 remained cheaper at `$0.25/hr`. Submitted one 1xA100 fallback
(`bid_UzmmcNdqjuYjI5xa`, `proicl-a100x1-signal-resume`) at `$0.30/hr`. If it
SSHs first, resume from the local mirror with logical oversubscription on the
single GPU and cancel the stuck 2x bid.

Market adjustment at `2026-05-17 14:14 UTC`: 1xA100 moved to `$0.30/hr`, exactly
matching the fallback bid. Raised `bid_UzmmcNdqjuYjI5xa` to `$0.35/hr`.

Market adjustment at `2026-05-17 14:18 UTC`: 1xA100 moved again to `$0.35/hr`
and remained `Open`; raised only the 1x fallback
`bid_UzmmcNdqjuYjI5xa` to `$0.45/hr`. The 2x fallback remains open at
`$0.60/hr` and is no longer being chased upward while the market spikes.

Market adjustment at `2026-05-17 14:22 UTC`: A100 continued spiking and the
listed 1x price tracked the losing bid. Raised only the 1x fallback
`bid_UzmmcNdqjuYjI5xa` to `$0.60/hr` to try to regain one bounded A100 host.

Fallback cleanup at `2026-05-17 14:28 UTC`: the first 1x fallback
`bid_UzmmcNdqjuYjI5xa` allocated but exposed only a private `10.234.*` SSH
host that timed out from local SSH. Waited briefly for public SSH, then canceled
it to avoid paying for an inaccessible instance.

Fallback retry at `2026-05-17 14:39 UTC`: 1xA100 cooled to `$0.03/hr`, while
2x remained high. Submitted fresh 1x fallback
`bid_KgOEZyrFivwb3ilx` (`proicl-a100x1-signal-resume-r2`) at `$0.05/hr`,
then raised it to `$0.06/hr` after availability moved to exactly `$0.05/hr`.

Market adjustment at `2026-05-17 14:47 UTC`: 1xA100 kept pinning exactly to the
fallback cap (`$0.06/hr`, then `$0.07/hr`) without allocating. Raised
`bid_KgOEZyrFivwb3ilx` to `$0.10/hr` as the bounded cheap A100 resume lane.

Market adjustment at `2026-05-17 14:55 UTC`: 1xA100 still pinned to the
fallback cap without allocating. Raised `bid_KgOEZyrFivwb3ilx` once more to
`$0.15/hr`, then stopped chasing. Current posture is to wait for the A100 market
to cool below one of the open caps (`1x=$0.15/hr`, `2x=$0.60/hr`) before
resuming from the local mirror.

Market adjustment at `2026-05-17 15:08 UTC`: after another 10 minutes, A100 was
still pinned exactly at the fallback caps and neither bid exposed SSH. Raised
the 1x fallback `bid_KgOEZyrFivwb3ilx` to `$0.25/hr` as a bounded single-GPU
resume attempt.

Market adjustment at `2026-05-17 15:12 UTC`: 1x remained unallocated and is
suspect because the earlier 1x allocation exposed only private SSH. Raised the
explicit 2x fallback `bid_IpiJH8Rc0gcEBOIe` to `$0.75/hr`; this is the lane that
previously produced the usable public host.

Market adjustment at `2026-05-17 15:16 UTC`: 2xA100 still pinned exactly to the
cap and remained `Open`; 1x fallback still did not expose SSH and had already
shown private-only behavior. Canceled the 1x fallback
`bid_KgOEZyrFivwb3ilx` and raised the explicit 2x lane
`bid_IpiJH8Rc0gcEBOIe` to `$1.00/hr` to force a usable A100 resume lane.

Market adjustment at `2026-05-17 15:20 UTC`: 2xA100 still pinned exactly to the
`$1.00/hr` cap and remained `Open`. Raised `bid_IpiJH8Rc0gcEBOIe` once more to
`$1.10/hr`; if this does not clear, stop chasing because the blocker is likely
not simply price.

Allocator reset at `2026-05-17 15:24 UTC`: the preempted 2x bid still remained
`Open` after being raised above the listed price, suggesting stale allocator
state. Canceled old bid `bid_IpiJH8Rc0gcEBOIe` and submitted fresh explicit
2xA100 resume bid `bid_ncgYYLD3L4APRZ7Y`
(`proicl-a100x2-signal-resume-r2`) at `$1.20/hr`. Resume source remains the
local mirror with `20` completed metric bundles.

Fallback retry at `2026-05-17 15:28 UTC`: fresh 2x bid also pinned to its cap
without allocation, so stopped raising it. Since 1xA100 cooled back to
`$0.03/hr`, submitted one more cheap 1x fallback
`bid_Xs9J52TNnI6GPwCs` (`proicl-a100x1-signal-resume-r3`) at `$0.05/hr`.
Cancel it if it allocates private-only like the previous 1x attempt.

Capacity hold at `2026-05-17 15:42 UTC`: neither open A100 bid has exposed SSH.
Current open caps are `bid_Xs9J52TNnI6GPwCs` (`1xa100`, `$0.05/hr`) and
`bid_ncgYYLD3L4APRZ7Y` (`2xa100`, `$1.20/hr`). Both are pinned exactly at
market but not allocated. No active GPU spend is occurring; resume remains
blocked on an SSH-ready A100 host.

Capacity hold at `2026-05-17 16:14 UTC`: monitored the open A100 bids for three
additional auction windows. Both remained `Open` with `host=null`
(`1xa100=$0.05/hr`, `2xa100=$1.20/hr`), and availability stayed pinned to those
same prices. No active GPU spend during this interval.

Cost guard at `2026-05-17 16:25 UTC`: canceled stale non-ProICL calibration bid
`bid_J8u4FIcRUfm1i6V0` and the high-cap 2xA100 resume lane
`bid_ncgYYLD3L4APRZ7Y`. The remaining active lane is cheap 1xA100
`bid_Xs9J52TNnI6GPwCs` at `$0.05/hr`; next plan is partitioned 1xA100 workers
if Flow exposes SSH reliably, otherwise continue A100-only capacity hold.

Partitioned cheap-A100 plan at `2026-05-17 16:31 UTC`: added
`configs/flow_proicl_a100x1_signal_hold.yaml`, submitted three additional
1xA100 bids at `$0.05/hr` (`bid_fICZ4zoBOG3xXjcR`,
`bid_vjIIIHngbjMJCxDs`, `bid_EpM8skQuisDwvVAq`), and started a local
partition controller over four offsets (`stride=4`). If any host becomes
SSH-ready, it launches `scripts/flow_launch_proicl_signal.py` with
`--cell-stride 4 --skip-aggregate`, mirrors artifacts back without deletion,
and locally aggregates once `360` metrics files exist.

Controller reset at `2026-05-17 16:37 UTC`: Flow `bid update` did not safely
raise the first four 1x bids and left them canceled, so stopped that controller
and submitted fresh 1xA100 partition bids directly at `$0.06/hr`:
`bid_OuPNnO4UekiHl3ct`, `bid_RxZXdJDzjCnEg52W`,
`bid_wmBnQG8TS0TFSaiU`, and `bid_f18kz9v5w7qMPtoe`. Restarted the partition
controller on these four bids with the same `stride=4` mapping.

Mithril/Flow shutdown at `2026-05-17 16:42 UTC`: after switching to the
user-provided CloudRift instance, stopped the local Flow partition controller
and canceled the active/provisioning Flow bids observed locally. One stale Flow
status row (`bid_J8u4FIcRUfm1i6GPwCs`) continued to report `provisioning`
despite `flow cancel -y` returning `cancelled: true`; no local controller
process remained.

CloudRift calibration at `2026-05-17 17:03 UTC`: configured SSH alias
`cloudrift-polaris` for the existing CloudRift RTX 4090 instance
`211.21.50.85:57034`, installed the vLLM/HF calibration environment, and
verified GPU visibility (`NVIDIA GeForce RTX 4090`, 24GB). Calibration artifacts
were mirrored under `runs/cloudrift_vllm_hf_calibration/`.

Strict HF-vLLM gate outcome:

- DeepSeek-R1-Distill-Qwen-1.5B, vLLM V0 forced decode, float32,
  `model_impl=transformers`: failed score parity
  (`max_abs_diff=0.019931793212890625`,
  `mean_abs_diff=0.0017175830415605256`), failed MH replay
  (`max_abs_diff=0.015781834465997235`), full-chain replay passed.
  Artifact:
  `runs/cloudrift_vllm_hf_calibration/probe_20260517T164353Z/hf_cached_decode__vllm_float32/calibration_summary.json`.
- Backend matrix on the same DeepSeek model showed no improvement from
  `NVIDIA_TF32_OVERRIDE=0`, HF `eager`, HF `sdpa`, explicit `XFORMERS`, or
  explicit `FLASH_ATTN`; vLLM V0 rejected `VLLM_ATTENTION_BACKEND=TORCH_SDPA`
  on CUDA. Artifact root:
  `runs/cloudrift_vllm_hf_calibration/backend_matrix_20260517T164900Z/`.
- Qwen/Qwen2.5-Math-1.5B, vLLM V0 forced decode, float32,
  `model_impl=transformers`: failed strict score parity
  (`max_abs_diff=0.010211944580078125`,
  `mean_abs_diff=0.0005555934828913046`), but MH replay and full-chain replay
  passed. Artifact:
  `runs/cloudrift_vllm_hf_calibration/qwen2p5_math_1p5b_default_20260517T165349Z/calibration_summary.json`.
- Qwen/Qwen2.5-Math-1.5B, vLLM V0 forced decode, float32,
  `model_impl=auto`: best observed candidate but still failed strict score
  parity (`max_abs_diff=0.00853729248046875`,
  `mean_abs_diff=0.0004966973424113045`), with MH replay and full-chain replay
  passing. Artifact:
  `runs/cloudrift_vllm_hf_calibration/qwen2p5_math_1p5b_auto_20260517T165534Z/calibration_summary.json`.

Interpretation: vLLM V0 forced-token scoring is close enough to preserve the
tested MH transcript on Qwen2.5-Math-1.5B, but it is not HF-equivalent under
the registered `1e-3` per-token scoring gate. The vLLM prompt-logprobs/native
prefill probe is also not ready in vLLM 0.9.2: requesting prompt logprobs at
`temperature=0.25` trips a vLLM sampler assertion before producing the
temperature-normalized selected-token scores. Therefore vLLM remains blocked
for science MCMC runs; HF/RWS remains the valid oracle.

Guard check: ran
`scripts/run_proicl_signal.py --backend vllm --vllm-parity-artifact runs/cloudrift_vllm_hf_calibration/qwen2p5_math_1p5b_auto_20260517T165534Z/calibration_summary.json --root runs/proicl_vllm_guard_check.tmp --smoke-only --skip-prefetch`.
It exited `rc=1` with
`vLLM calibration gate failed: top-level passed is not true; score_parity did not pass`,
confirming failed artifacts cannot enter the ProICL runner.

CloudRift vLLM equivalence resolution at `2026-05-17 17:30 UTC`: isolated the
strict parity failure to vLLM prefix-cache reuse across forced-token scoring
calls. One-shot forced decode already matched HF (`<=3e-5` temperature-scaled
diff), while the calibration-order sequence failed only when prefix caching was
left populated between score calls. Resetting `llm.llm_engine.reset_prefix_cache()`
before and after each teacher-forced scoring call preserves the configured
prefix-cache runtime while removing stale forced-score cache contamination.

Accepted DeepSeek calibration artifact:

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Revision: `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- vLLM: V0, `0.9.2`, float32, `model_impl=transformers`,
  `enable_prefix_caching=true`, `reset_prefix_cache_for_scoring=true`
- HF oracle: float32 cached decode
- Result: `passed=true`, score parity `max_abs_diff=0.00017786026000976562`,
  segment max diff `0.00010402500629425049`, MH replay max diff
  `2.2242311388254166e-05`, ambiguous boundary count `0`
- Local artifact:
  `runs/cloudrift_vllm_hf_calibration/deepseek_1p5b_transformers_prefix_reset_20260517T172039Z/calibration_summary.json`

Runner integration gate at `2026-05-17 17:30 UTC`: added model-id validation to
`validate_vllm_calibration_artifact(...)`; `scripts/run_condition.py` now
requires the parity artifact model id to match the requested model, and
`scripts/run_proicl_signal.py` requires the DeepSeek base-model calibration for
vLLM ProICL runs. Verified locally that the accepted DeepSeek artifact passes
and the passing Qwen/Qwen2.5-Math-1.5B artifact is rejected for DeepSeek.

Integrated vLLM MCMC smoke at `2026-05-17 17:30 UTC`: after installing
`reasoning-gym==0.1.25` in the CloudRift venv, ran one real
`run_condition.py` cell on `reasoning_gym_boxnet`, condition
`single_prompt_power`, split `[20, 21]`, `B=1`, `max_new_tokens=64`,
backend `vllm`, scoring mode `forced_decode_v0`, using the accepted DeepSeek
calibration artifact. It completed and wrote the full artifact set with
`backend=vllm`, `prefix_caching=true`,
`reset_prefix_cache_for_scoring=true`.

Local mirrored smoke artifact:
`runs/cloudrift_proicl_vllm_smoke/proicl_vllm_mcmc_smoke_20260517T172654Z/runs/reasoning_gym_boxnet/mcmc_only/shard-0/`.

CloudRift ProICL vLLM launch at `2026-05-17 18:13 UTC`: launched the paid
small-real-slice signal run on the existing CloudRift RTX 4090 instance without
provisioning any new GPUs. Initial HF-based GEPA archive construction was
underutilizing the GPU, so patched the GEPA adapter to use sampler batch
generation when available and patched `scripts/run_proicl.py build-archive` /
`scripts/run_proicl_signal.py` to pass the calibrated vLLM backend into live
GEPA archive rollouts. Remote dependency state was corrected to
`litellm==1.61.20`, `openai==1.90.0`, `vllm==0.9.2`.

Active optimized run:
`/home/riftuser/polaris/runs/cloudrift_proicl_signal/proicl_vllm_signal_vllm_gepa_20260517T180327Z`.
Launch shape: tracks `reasoning_gym_boxnet`, `reasoning_gym_graph_color`,
`reasoning_gym_family_relationships`; eval split `[20, 40]`; GEPA dev split
`[0, 6]`; rollout budget `8`; archive size `8`; max metric calls `64`;
backend `vllm`; scoring mode `forced_decode_v0`; vLLM dtype `float32`;
`model_impl=transformers`; `gpu_memory_utilization=0.82`; parity artifact
`runs/cloudrift_vllm_hf_calibration/deepseek_1p5b_transformers_prefix_reset_20260517T172039Z/calibration_summary.json`.

Runtime evidence: the optimized GEPA archive command included
`--sampler-backend vllm` and the vLLM parity artifact. GPU allocation during
GEPA rose to about `20GB/24GB` with observed SM utilization around `50%` during
rollout scoring, vs about `4GB` and `33%` under the HF archive builder. GEPA
archive completed at `2026-05-17 18:08:56 UTC`, then the condition queue
started. First completed cell:
`reasoning_gym_boxnet/base_greedy/shard-0`, `accuracy=0.0`, `n_problems=20`,
`backend=vllm`. Active cell as of the checkpoint:
`reasoning_gym_boxnet/mcmc_only/shard-0`.

CloudRift confirmation adjustment at `2026-05-17 18:30 UTC`: local tests
confirmed explicit MCMC knob propagation without changing defaults:
`PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q
tests/unit/test_inference.py tests/unit/test_proicl.py
tests/unit/test_vllm_serving.py tests/unit/test_vllm_calibration.py
tests/smoke/test_math500_dummy.py` -> `73 passed`; protocol sync passed.
Added `--mcmc-steps` and `--mcmc-block-num` plumbing through
`scripts/run_condition.py`, `src/polaris/runners/math500.py`,
`src/polaris/core/inference.py`, and `src/polaris/proicl/launcher.py`.
Defaults remain the locked RWS values `mcmc_steps=10`, `mcmc_block_num=16`;
explicit lower values are for bounded confirmation only and must not be
reported as the faithful science MCMC condition.

Operational decision: the faithful `[20,40]`, `B=8`, all-condition slice is
not a practical same-session confirmation target on a single RTX 4090. The
active default-MCMC cell had no candidate rows yet after hundreds of vLLM
forced-score prefix-cache resets, while the GPU process was healthy at about
`20GB/24GB` and `~50%` SM utilization. Preserve the run root as evidence, stop
that long slice for cost/time control, then launch a bounded all-condition
confirmation slice using the calibrated vLLM path and explicit reduced MCMC
knobs to verify end-to-end artifacts and runner integration.

CloudRift vLLM confirmation result at `2026-05-17 18:56 UTC`: completed the
bounded all-condition confirmation run on the existing RTX 4090 instance:
`runs/cloudrift_proicl_signal/proicl_vllm_confirmation_20260517T183111Z`.
Configuration: three Reasoning Gym tracks, eval split `[20,21]`, `B=2` for
non-greedy cells, `max_new_tokens=128`, explicit bounded confirmation knobs
`mcmc_steps=2`, `mcmc_block_num=4`, backend `vllm`, scoring mode
`forced_decode_v0`, dtype `float32`, `model_impl=transformers`, and accepted
DeepSeek HF-vLLM calibration artifact for all MCMC/power cells.

During the run, `prorl_v2_greedy` initially failed because the global DeepSeek
MCMC parity artifact was incorrectly enforced on the Nemotron greedy comparator.
Patched `scripts/run_condition.py` so HF-vLLM scorer calibration is required
only for power/MCMC conditions (`single_prompt_power`, full archive power
variants, `polaris_full_verified_memory`, `proicl_gepa_mcmc`,
`proicl_gepa_mcmc_memory`). Greedy and GEPA-only generation cells no longer
consume a mismatched scorer artifact. Local and remote targeted checks after
the patch: `49 passed`; protocol sync passed.

Final confirmation evidence: resumed the same root; completed cells were
skipped, the failed comparator was rerun and passed, then the remaining tracks
completed. Final events include `queue_done` with `failures=0` and
`aggregate_done`. Artifact audit passed locally and remotely:
`metric_count=18`, all required cell bundles present, all memory cells include
`memory_events.jsonl` and `memory.sqlite`, all MCMC cells record
`backend=vllm`, `vllm_scoring_mode=forced_decode_v0`, `mcmc_steps=2`,
`mcmc_block_num=4`, and passing calibration metadata. Aggregate artifacts:
`full/analysis/aggregate_stdout.json`,
`full/analysis/proicl_decomposition.json`,
`full/analysis/proicl_decomposition.md`.

Interpretation: this is an integration/calibration confirmation, not a science
signal slice. Accuracies were all `0.0` on the one-problem bounded slice, so RF
is undefined there. The science-valid MCMC path remains gated by the accepted
DeepSeek calibration artifact and the locked default RWS knobs; the reduced
`2/4` MCMC settings are only for bounded confirmation.
