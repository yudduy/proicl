# ProRL Recovery Audit

Status: multi-resource live scheduler is active; CloudRift is fallback, not the
primary target.  
Goal: complete the standalone recoverable-fraction audit through resource
probes, cost gates, Phase 0 replication, filtered Phase 2/3 artifacts, or stop
at the first hard gate failure.

## Question

For public ProRL/BroRL checkpoints, what fraction of the gain over the frozen
base can be recovered by frozen-base inference, and what residual appears to
require weight updates?

Recoverable fraction:

```text
RF = (A_frozen_inference - A_base) / (A_trained_checkpoint - A_base)
```

The value is clipped to `[0, 1]`; it is undefined if the trained checkpoint
does not exceed the base.

## Fixed Checkpoints

- Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
  (`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`)
- ProRL v1: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@v1`
  (`b89048893f95246c6b5749b287f0049e6df42ee9`)
- ProRL v2: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@main`
  (`c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0`). The public label is `main`;
  artifact ETags are recorded in the registry because `main` and `v2` currently
  point to different refs with matching config and weight ETags.
- BroRL: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@brorl`
  (`3441fcdf8c6e81a2959e6352ff50122e3c677d72`)

## Tracks

Start with MATH500, GPQA-Diamond, and Reasoning Gym `boxnet`,
`graph_color`, and `family_relationships`. Code is deferred until the
mechanism study is clean.

GPQA-Diamond remains gated by the existing HF/auth policy. Oracle answer keys
are offline evaluators only.

## Phase Gates

Phase 0 is external replication. Before any RF claim, reproduce Karan-Du RWS on
`Qwen/Qwen2.5-Math-7B`, MATH500, `alpha=4`, `max_new_tokens=3072`,
`block_num=16`, `N_MCMC=10`. Target accuracy is `0.748` with tolerance
`+-0.02`. Internal HF/vLLM parity is not a substitute.

Phase 1 establishes the ProRL/BroRL gap: base, ProRL v1, ProRL v2, BroRL;
record pass@1/pass@16/pass@128, response length, verifier result, token ids,
backend, and one canonical `phase1_results.parquet`.

Phase 2 runs only on the ProRL-only denominator set derived from
`phase1_results.parquet`: problems where ProRL v2 or BroRL solves and the base
does not. If a task has no trained-checkpoint improvement, RF is undefined for
that task and Phase 2/3 are skipped. The frozen-base ladder is greedy, BoN
`K={4,16,64,256,1024}` at `T=1` and `T=1.2`, RWS MCMC, mixed-alpha MCMC, GEPA archive,
archive+mixed-alpha, archive+mixed-alpha+verified memory.

Phase 3 launches only after Phase 1 and Phase 2 rung-7 artifacts exist. It
derives `phase3_input_set.parquet` once: ProRL/BroRL solves, base rung-7 fails.

## Locked Bucket Rules

Bucket assignment is ordered:

1. `search_limited`: ProRL trace normalized base logprob is in the top quartile
   of base-generated trajectory normalized logprobs for the same task family.
2. `prompt_conditional`: a prompt variant solves without memory transplant.
3. `memory_conditional`: transplant pass@16 minus control pass@16 is at least
   10 percentage points.
4. `weight_only`: none of the above.

Control for the transplant test is a random unrelated trajectory of similar
token length from a different task family.

## Logprob Authority

Phase 3.1 bucket assignment uses HF/RWS-style
`score_segments(..., temperature=1.0)` as the authority. vLLM can generate and
can provide forced-score diagnostics only after a full-trajectory parity smoke;
native vLLM `logprobs` / `prompt_logprobs` is not accepted for assignment.

## Resource Contract

Use `configs/prorl_live_resources.json` as the launch profile source:

- `farmshare_l40_free`: free continuation for shardable Phase 0/1 work.
- `flow_a100_weekend`: capped 4x A100 80GB accelerator; default max bid
  `$0.025/GPU-hr`, preferred 4x total price `<= $0.10/hr`, initial spend cap
  `$10`.
- `modal_burst`: Phase 3/debug bursts only; initial cap `$25`.
- `cloudrift_fallback`: RTX 4090 fallback if FarmShare and Flow block; initial
  cap `$25`.

Every paid launch still needs artifact directory, cache path, split, seed,
model, backend, cost estimate, cost cap, and explicit user authorization.

```text
HF_HOME=/workspace/.cache/huggingface
HF_HUB_CACHE=/workspace/.cache/huggingface/hub
POLARIS_REPO_DIR=/workspace/polaris
POLARIS_RUN_ROOT=/workspace/polaris/runs/prorl_recovery
```

Use one persistent model cache. Do not redownload base/ProRL refs per rung.

GEPA reflection uses xAI through an OpenAI-compatible/LiteLLM adapter:

```text
XAI_BASE_URL=https://api.x.ai/v1
XAI_REFLECTION_MODEL=grok-4.3
XAI_REFLECTION_INITIAL_CAP_DOLLARS=30
XAI_REFLECTION_HARD_CAP_DOLLARS=100
```

`XAI_API_KEY` must stay in ignored local `.env` files or backend secret stores.
Artifacts must record model, token counts, cost estimate, and response hashes,
but never the key.

## Token Cap Policy

Lock caps before any 100-problem run:

```text
Phase 0 Karan-Du: 3072
MATH500: 4096
GPQA-Diamond: 2048
Reasoning Gym: 8192
```

If a smoke has cap-hit rate above 5%, double that task cap once, re-run smoke,
and lock the result before main.

## Smoke Ladder

1. Resource probe: `nvidia-smi`, CUDA/PyTorch/vLLM imports, HF cache
   write, snapshot download for base and ProRL refs.
2. xAI reflection dry smoke with redacted manifest.
3. Tiny smoke: 1 MATH500 problem x base x pass@2 and 1 Reasoning Gym problem x
   base x pass@2, writing full artifacts and wall/token stats.
4. Cost projection gate; stop if projected Phase 1 exceeds the approved cap.
5. Phase 0 Karan-Du ladder: 1 problem, 20 problems, full 500 only if cost is
   acceptable.
6. Phase 1 denominator, then filtered Phase 2/3 only after gates pass.

## Active `/goal`

Complete the multi-resource ProRL Recoverable Fraction audit without stopping
until cost gates, main artifacts, and audits are complete, or until a hard gate
fails.
Maintain protocol rigor: Phase 0 Karan-Du replication gate before RF claims; no
GPQA official rows unless HF/auth or `GPQA_DIAMOND_PATH` is present; no ProRL
traces in the RF numerator memory; and write progress after every phase.
