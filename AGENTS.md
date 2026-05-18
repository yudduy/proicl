# AGENTS.md

## Identity

Thinking partner, not content generator. First principles over opinions. Compress
ruthlessly; every word must carry weight.

Use this source order:

1. Current checked-out files and run artifacts.
2. `PROPOSAL.md` for the scientific contract.
3. `TODO.md` for the operational contract.
4. `runs/progress.md` for observed evidence and decisions.
5. Primary papers, upstream source repos, then docs.

If the three protocol files conflict, stop execution and resolve the conflict
with a dated `runs/progress.md` entry before starting any new experiment.

## Current Contract

Protocol: `POLARIS-v3.1`.

POLARIS asks which RL-style reasoning gains can be recovered from a frozen base
model through inference-time composition:

- prompt-archive spread;
- diversity-preserving sharpening;
- verifier-gated memory.

Current active work is **publishable end-to-end experiment readiness**, not a
final science run. The repository must support official MATH500, HumanEval+,
and GPQA-Diamond locks; generic condition execution; full POLARIS verified
memory; baseline conditions; archive-memory construction; artifact auditing;
and cost/data gates before any paid or final launch.

The readiness bar is proposal-wide: every `PROPOSAL.md` experiment family should
have a local executable smoke and artifact bundle unless a real external blocker
is proven. Repo-local missing code or `NotImplementedError` is work, not a
valid deferral.

Hard blocks:

- No full final run, Modal/Mithril/Flow bulk generation, or paid model call
  without a fresh explicit user command and cost cap.
- No paid run without artifact directory, cache path, split, seed, model,
  backend, cost estimate, cost cap, and explicit per-launch user authorization.
- No GPQA official rows without accepted Hugging Face terms plus
  `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`, or an official local
  `GPQA_DIAMOND_PATH` cache from that account. Do not write GPQA examples into
  tracked docs.
- No protocol change unless mirrored across `PROPOSAL.md`, `TODO.md`, and
  `runs/progress.md` in the same work session.
- The completed `C0c` run is evidence, not authorization for old Phase 1/2.

Current implementation state:

- `polaris_full_verified_memory` is the publishable-method condition:
  descriptor-aligned archive + mixed power sampling + verifier-gated persistent
  memory + GEPA/archive-memory build provenance.
- MATH500 and HumanEval+ split locks are cached locally for dev,
  `small_real_slice`, and final. GPQA-Diamond split locks are `pending_auth`
  until HF auth or official local cache is configured.
- Power sampling, archive+power, DC, ACE, GEPA-only, P2O ledger, and GRPO
  ledger remain baselines or references; do not collapse them into the POLARIS
  method claim.

Known run state:

- Completed: full-164 official RWS MCMC-only `C0c` reproduction on
  `Qwen/Qwen2.5-7B` HumanEval.
- Valid local summary: `task_json_count=164`, duplicate task indices `[]`,
  HumanEval `91/164 = 0.5549`, HumanEval+ `79/164 = 0.4817`.
- Flow/Mithril active 4xA100 bid `polaris-a100x4-ckpt-66304a` is cancelled;
  8x bid `polaris-a100x8-ckpt-e8b2d2` is paused/non-running per `TODO.md`.

## Repo Map

This directory is not a git repository. Treat the filesystem snapshot as source
of truth and avoid destructive edits.

- `PROPOSAL.md`: scientific thesis, references, execution governance,
  R5 infrastructure contract, drift-lock protocol.
- `TODO.md`: live operational state, pre-registration, blockers, phase order.
- `runs/progress.md`: observed evidence and dated decisions.
- `runs/rws_official_full164_mcmc_ckpt/`: completed C0c artifacts and EvalPlus
  recompute.
- `runs/readiness_smoke.tmp/`: generated local readiness smoke artifacts.
- `scripts/check_protocol_sync.sh`: protocol drift guard.
- `scripts/smoke_polaris_readiness.py`: CPU-only full local readiness smoke.
- `scripts/plan_production_runs.py`: production run graph writer.
- `scripts/run_condition.py`: typed track/condition runner entry point.
- `scripts/run_math500.py`: compatibility shim over the generic runner.
- `scripts/lock_datasets.py`: official split-lock writer.
- `scripts/build_polaris_archive.py`: archive-memory co-construction entry point.
- `scripts/analyze_results.py`: artifact-only final analysis.
- `scripts/modal_app.py`: bounded HF/SGLang/Modal smokes.
- `scripts/modal_vllm_app.py`: isolated vLLM Modal smokes.
- `scripts/build_mapelite.py`: MATH500 archive construction.

Package layout:

- `src/polaris/config.py`: locked defaults, model registry, track registry.
- `src/polaris/core/archive.py`: `FrozenArchive`, `PromptEntry`,
  `MATH500_ARCHIVE_V1`.
- `src/polaris/core/descriptor.py`: versioned trace descriptor extractors.
- `src/polaris/core/inference.py`: archive allocation, sampling, verifier
  selection, optional memory and trajectory cache.
- `src/polaris/core/memory.py`: verifier-gated memory, distillation, leakage
  screen, Beta-Bernoulli reliability.
- `src/polaris/core/mapelite.py`: prompt-grid MAP-Elites.
- `src/polaris/core/mixed_alpha.py`: fixed, mixed, and decaying alpha schedules.
- `src/polaris/infra/preflight.py`: paid/scale-capable launch gate.
- `src/polaris/infra/mcmc.py`: HF batched MCMC correctness oracle mechanics.
- `src/polaris/infra/serving/hf.py`: RWS/HF oracle backend.
- `src/polaris/infra/serving/vllm.py`: vLLM V0 optimized MCMC candidate.
- `src/polaris/infra/serving/sglang.py`: SGLang client; MCMC scoring blocked.
- `src/polaris/io/trajectory_cache.py`: SQLite trajectory cache.
- `src/polaris/runners/math500.py`: current fully wired condition runner.
- `src/polaris/runners/condition_runner.py`: track-generic future entry point;
  currently re-exports MATH500 runner.
- `src/polaris/evals/`: MATH500, HumanEval+, GPQA loaders and verifiers.
- `src/polaris/io/artifact_audit.py`: smoke, small-real-slice, and final
  artifact gate.
- `src/polaris/io/dataset_locks.py`: split-aware lock validation.
- `src/polaris/stats/`: bootstrap, McNemar, factorial, break-even helpers.
- `src/polaris/vendored/`: runtime-copied RWS, GEPA, EvalPlus, Dynamic
  Cheatsheet code. Edit only with source evidence.
- `upstream/`: read-only reference clones. Use for source checking and
  copy-before-adapt, not direct runtime imports.

Reference clone commits observed at init:

- RWS `upstream/reasoning-with-sampling`: `720a8e9d084c87a630595e316f5260f1d7c3446c`
- GEPA `upstream/gepa`: `ce51b50cd196b539c25fae99ad0e0255c23004a4`
- EvalPlus `upstream/evalplus`: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- Dynamic Cheatsheet `upstream/dynamic-cheatsheet`: `5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9`
- vLLM `upstream/vllm-v0.9.2`: `a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f`
- SGLang `upstream/sglang-v0.4.7`: `4f723edd3baf3823eddfb9d6426548daba17c687`

## Verification Commands

Prefer the repo venv:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/
bash scripts/check_protocol_sync.sh
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/plan_production_runs.py --out runs/production/plan.json
```

The readiness report should end with `passed=True`, `deferred_tracks=[]`,
`deferred_experiments=[]`, and cache replay `generation_calls_on_replay=0`.

Targeted docs/protocol verification:

```bash
bash scripts/check_protocol_sync.sh
```

Full POLARIS preflight only:

```bash
./.venv-eval/bin/python scripts/build_polaris_archive.py \
  --track math500 \
  --mode freeze \
  --dev-split 75 76 \
  --out runs/archive_build_smoke.tmp \
  --dry-run

./.venv-eval/bin/python scripts/run_condition.py \
  --track math500 \
  --model-key qwen2.5-math-7b \
  --condition polaris_full_verified_memory \
  --archive runs/archive_build_smoke.tmp/archive.json \
  --split 0 1 \
  --out runs/preflight_full_polaris.tmp \
  --polaris-source-hash <hash> \
  --preregistration-anchor TODO.md#polaris-full \
  --preflight-only
```

Legacy MATH500 paid-run preflight only:

```bash
./.venv-eval/bin/python scripts/run_math500.py \
  --condition full_archive_mixed \
  --archive runs/2026-05-12-polaris-math500-v1/archive.json \
  --test-slice 0 75 \
  --seed 17 \
  --polaris-source-hash <hash> \
  --preregistration-anchor TODO.md#polaris-math500-v1 \
  --out runs/2026-05-12-polaris-math500-v1/full_archive_mixed/seed-17 \
  --trajectory-cache runs/2026-05-12-polaris-math500-v1/trajectories.sqlite \
  --estimated-dollar-cost <estimate> \
  --cost-cap-dollars <cap> \
  --user-authorized-paid-run \
  --preflight-only
```

Bounded Modal commands are allowed only after local gates pass and the user gives
explicit launch authorization. Use the venv Modal CLI; the global Homebrew
`modal` shebang was previously broken.

Known accepted bounded vLLM smoke shape:

```bash
./.venv-eval/bin/python -m modal run scripts/modal_vllm_app.py::smoke_vllm_batched_power_path \
  --batch-size 8 \
  --max-new-tokens 512 \
  --block-num 4 \
  --mcmc-steps 1 \
  --estimated-dollar-cost 0.25 \
  --cost-cap-dollars 0.50 \
  --user-authorized-paid-run
```

## Engineering Rules

- Test-first for bugs: write a failing test, fix, then run the exact test.
- Never delete, skip, or weaken tests to make a change pass.
- Minimal diffs only. No opportunistic refactors.
- Search before create. Prefer existing module patterns over new abstractions.
- Do not mark repo-local missing surfaces as deferred; build the smallest local
  smoke path unless the blocker is genuinely external.
- When porting from a reference repo, copy the source first, then adapt.
- Treat `upstream/` as read-only. Runtime code lives under `src/polaris/`.
- Do not scan or edit `.venv-eval/` except to run tools from it.
- Keep generated artifacts out of commits if this directory is later made a repo.

## Research Rules

- Literature before proposal. If an experiment idea changes, search papers first.
- Verify paper/mechanism claims against primary sources.
- Use upstream source and DeepWiki before claiming how a library works.
- alphaXiv can find papers, but mechanism details must be checked against the
  paper text.
- For math derivations, use SymPy/SciPy and numerical sweeps; do not rely on
  prose derivations.

## POLARIS Invariants

- Base model stays frozen in core experiments.
- MATH500 v1 model is `Qwen/Qwen2.5-Math-7B`; HumanEval C0c used
  `Qwen/Qwen2.5-7B`. Do not carry model choices across benchmarks.
- Research-faithful split policy: optimizer development uses non-final pools;
  final reporting uses the full official benchmark. Current locks are MATH dev
  `DigitalLearningGmbH/MATH-lighteval train[0:500]`, MATH final
  `MATH500 test[0:500]`; code dev `MBPP+ [0:378]`, HumanEval+ final
  `test[0:164]`; GPQA dev `gpqa_main_minus_diamond[0:250]`, GPQA-Diamond
  final `train[0:198]`.
- Final POLARIS method condition is `polaris_full_verified_memory`.
- Default full-method memory mode is `distilled_strategies`; raw traces and
  cross-task memory are ablations.
- Locked MCMC defaults: `MCMC_STEPS=10`, `MCMC_BLOCK_NUM=16`,
  `MAX_NEW_TOKENS=3072`, temperature `1 / alpha`.
- `alpha=1` may short-circuit to plain temperature-1 sampling.
- `full_archive_mixed` alternates alpha `(4.0, 1.0)` by sample-index parity.
- `full_archive_decaying` cycles `(4.0, 3.0, 2.0, 1.0)`.
- Selector is argmax verifier score with first-wins tie break.
- Required artifacts per production run: `manifest.json`, `archive.json`,
  `candidates.jsonl`, `scores.jsonl`, `selected.jsonl`, `metrics.json`,
  `costs.json`, `rollouts.json`, `preflight.json`, `environment.json`,
  `run_plan_cell.json`, and `audit.md`.
- Memory-enabled conditions must also write `memory_events.jsonl` and
  `memory.sqlite`. GEPA/archive conditions must write
  `archive_build_manifest.json`.
- `manifest.preregistration_anchor` must be non-empty and point to the locked
  protocol block.
- Cache keys are `(model_id, track, problem_id, prompt_id, sample_idx, alpha, seed)`.
- Cache hits must replay without generation calls and preserve verifier state.
- GPQA answer keys are offline evaluators only; selection must be non-oracle.
- Memory admission requires an independent verifier check and leakage screen.

## Serving Backend Policy

- HF/RWS is the correctness oracle because it exposes direct full-vocabulary
  logits for MH scoring.
- vLLM V0 is the accepted optimized candidate only in the source-checked
  configuration: `VLLM_USE_V1=0`, `dtype=float32`, `model_impl=transformers`,
  prefix caching enabled.
- Accepted vLLM MCMC path: native vLLM sampling, then batched forced-token
  scoring for `lp_norm` and `lp_unnorm`.
- vLLM fused sampling-recorder path is opt-in only and not scale-accepted.
- vLLM native `logprobs` / `prompt_logprobs` is not accepted for MH scoring.
- SGLang is not accepted for MCMC scoring until HF parity passes.
- Any optimized `score_segments` change needs HF parity under `1e-3` absolute
  difference before scale-up.

## Communication

Report evidence paths and exact commands. Say when a state is memory-derived or
from old artifacts. If blocked, name the missing contract field or failing gate.
