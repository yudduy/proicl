# POLARIS: Prompt-Organized Library of Archived Reasoning and Inference Strategies

Author(s): Duy (advisee), Mirac Suzgun (advisor)
Institution: Stanford
Type: Method proposal
Short name: POLARIS
Legacy names: CovComp; "POLARIS: Inference-Time Composition for RL-Style Reasoning Gains" (v2.0 working title)
Status: Revised conceptual framing locked; R5 infrastructure contract active
Audit date: 2026-05-13
Protocol version: POLARIS-v3.1
Last protocol update: 2026-05-13

Execution governance:
- `PROPOSAL.md` defines the scientific contract.
- `TODO.md` defines operational run state, owners, and phase sequencing.
- `runs/progress.md` records actual outcomes and signed decisions.

Drift rule:
- Any conflict among these three files blocks execution transitions.
- No locked protocol item may be changed without a protocol-amendment entry in all three files.

## Agent Execution Contract

This file is the project source of truth. Update it before changing the thesis, benchmark protocol, verifier policy, or contribution claim.

Cross-file invariants:
- `TODO.md` is binding for execution order, blockers, and ownership.
- `runs/progress.md` is binding for observed run evidence, results, and phase completion status.
- No agent may transition phases if TODO and progress disagree.

Every implementation agent must preserve these invariants:

1. Keep the base model frozen in all core POLARIS experiments.
2. Pre-register archive size, sample budget, allocation rule, alpha schedule, verifier, split, and statistical test before held-out evaluation.
3. Report uniform allocation before any adaptive allocation result.
4. Treat GPQA-Diamond answer keys as offline evaluators, not inference-time verifiers.
5. Separate exact-verifier results from LLM-judge or cross-model-consensus proxy results.
6. Store full candidate sets, verifier decisions, costs, prompts, retrieved memory entries, and selected outputs for every run.
7. Do not claim "RL replacement" unless the model, benchmark, protocol, and cost accounting are directly comparable to the cited GRPO-style reference.
8. No protocol amendment is valid unless it is recorded in `TODO.md` and `runs/progress.md` before additional execution.

Required run artifacts:

```text
runs/<date>-<slug>/
  manifest.json          # model, benchmark, split, commit/hash, config, seeds
  archive.json           # frozen prompt archive and descriptor statistics
  candidates.jsonl       # all generated candidates, prompt id, memory ids, sample metadata
  scores.jsonl           # verifier outputs and acceptance decisions
  costs.json             # archive-construction/inference/verifier/memory rollout + cost accounting
  metrics.json           # accuracy, oracle coverage, verifier coverage, latency, sample count, rollouts
  audit.md               # deviations, failures, false accepts, false rejects, notes
```

## 1. Conceptual Framing

A growing body of evidence suggests that some gains attributed to RL post-training on verifiable reasoning tasks can be reinterpreted as distribution sharpening over trajectories the base model already places nonzero probability mass on [R1-R3]. If this interpretation generalizes across a useful range of tasks, it becomes meaningful to ask: **how much of an RL post-trained model's behavior can be recovered by an inference-time procedure that produces sharpened, diverse, and verified outputs from a frozen base model?**

POLARIS is our attempt to find out, by composing three inference-time mechanisms.

### 1.1 Prompt Archive (inducing diverse conditional reasoning distributions)

A single prompt fixes one conditional over reasoning trajectories, so single-prompt sampling can only ever produce trajectories in that conditional's support. P2O frames this cleanly [R18]: a prompt `z` is a latent variable, and `pi(y | z, q)` has a different support than the baseline `pi(y | q)`. An archive `Z = {z_1, ..., z_k}` of behaviorally diverse prompts extends the accessible support by inducing several conditionals at once, which gives the verifier more chances to find an acceptable trajectory under at least one of them. The Coverage Principle [R4] makes the formal version of this argument for Best-of-N and test-time scaling.

We construct the archive using quality-diversity ideas. MAP-Elites [R10] is the original. AlphaEvolve [R17] and GEPA [R5] are the LLM-era versions. GEPA is closest to what we want; it matches or exceeds RL with ~35x fewer rollouts on its benchmarks.

### 1.2 Scalable Power Sampling under Each Conditional, with Diversity Preservation

Within each conditional, we use scalable power sampling to concentrate probability mass on high-likelihood trajectories. Reasoning with Sampling [R2] shows that MCMC-style sampling from a sharpened distribution `pi^alpha` approaches RL accuracy on MATH500, HumanEval, and GPQA without any weight updates. Scalable Power Sampling [R3] gives an autoregressive token-level approximation that runs at standard inference cost. Echo Chamber [R1] supplies the mechanistic story: RL on verifiable tasks appears to amplify behaviors already in the pretrained distribution, which is what sharpening would do.

Fixed-`alpha` sharpening has a known failure mode: it collapses within-conditional diversity. If we set `alpha = 4` on every sample, all `K` samples per prompt converge to roughly the same mode, and our effective per-prompt sample count drops toward 1. ProRL [R19] addresses a related problem in RL via entropy preservation (KL penalty, reference reset, decoupled clipping). Our inference-time analog is **mixed-alpha sampling** within each prompt's budget: some samples at `alpha = 4` (mode-seeking), some at `alpha = 1` (base distribution). This keeps within-conditional diversity while still getting the benefit of sharpening on the samples that need it.

### 1.3 Verifier-Gated Memory (accumulating transferable strategies across queries)

Dynamic Cheatsheet [R6] and ACE [R7] show that persistent test-time memory improves reasoning without weight updates. The catch is that both methods rely on the LLM itself to decide what to keep, which we expect to compound confidently-wrong entries over time. POLARIS replaces self-judgment with external verifier outcomes. A strategy gets admitted only if it passes an independent verifier check on the query where it was discovered, and reliability is tracked over later retrievals with a Beta-Bernoulli model.

### 1.4 Hypothesis

The hypothesis is that **spread + sharpening + accumulation** recovers a meaningful fraction of RL-style accuracy at substantially lower total compute than RL post-training itself. The empirical question is at what compute ratio, and on which task regimes.

The overarching question this proposal is built to answer:

> Can the capabilities that emerge from ProRL [R19] be recovered with pure inference techniques?

### 1.5 Standalone ProRL Recovery Audit Before Scale-Up

Before scaling the full POLARIS archive-memory experiment, we run a narrower
Recoverable Fraction audit on public ProRL/BroRL checkpoints. This audit asks
how much of the trained-checkpoint gain over the base model can be recovered by
frozen-base inference alone:

```text
RF = (A_frozen_inference - A_base) / (A_trained_checkpoint - A_base)
```

The audit is diagnostic. ProRL/BroRL traces must not seed POLARIS memory in the
main RF numerator. Any RF claim is blocked until an external Karan-Du/RWS
replication gate reproduces the published MATH500 power-sampling result on the
published setup (`Qwen/Qwen2.5-Math-7B`, `alpha=4`, `max_new_tokens=3072`,
`block_num=16`, `N_MCMC=10`, target accuracy `0.748 +- 0.02`). Internal
HF/vLLM parity is necessary engineering evidence, but not a substitute for
external replication.

## 2. Distinguishing from Existing Work

POLARIS is closest in spirit to P2O [R18]. They also use GEPA-style prompt evolution to crack samples that vanilla RL gradients miss, but they then distill the resulting trajectories into model weights via context distillation. POLARIS does the same prompt evolution but keeps the gains at inference time. We accumulate them in context-level memory rather than weights. Effectively: **"P2O without the distillation step."**

GEPA [R5] is the prompt-evolution machinery both methods inherit; we extend it with sharpening and verified memory at inference. DC [R6] and ACE [R7] use LLM self-judgment for memory curation; we use external verifiers. Reasoning-with-Sampling [R2] and Scalable Power Sampling [R3] show single-prompt sharpening can approach RL on individual benchmarks; we are asking whether composing sharpening with archive spread and verified memory does meaningfully better.

ProRL [R19] is the most important counterpoint. They show that prolonged RL with entropy preservation can expand the reasoning boundary beyond the base distribution, uncovering trajectories the base model genuinely cannot produce even at large `pass@k`. That defines the regime where we expect POLARIS to underperform (their Sustained regime: code, logic puzzles, OOD reasoning). The Diminished regime, where RL just sharpens (like most math benchmarks), is where we expect POLARIS to be competitive.

CycleQD [R8] applies quality-diversity at the parameter level via model merging, which is a different design space. Against RL itself, POLARIS keeps the model frozen. The comparison we care about is what fraction of the RL gain we can recover, and at what compute ratio.

## 3. Falsifiable Thesis

POLARIS tests whether a frozen base language model can recover a substantial fraction of RL-style reasoning gains on verifiable tasks by composing:

1. a bounded archive of behaviorally diverse prompts;
2. scalable power sampling under each prompt with diversity-preserving alpha schedules;
3. verifier-gated memory across queries.

POLARIS does not claim that reinforcement learning post-training is unnecessary in general. The claim is benchmark- and regime-specific: on reasoning tasks with reliable evaluators in the **Diminished** regime, a fixed prompt archive plus distribution sharpening plus verified memory may match or exceed selected GRPO-style reference baselines at substantially lower total rollouts, without updating model weights.

The thesis is falsifiable because the archive, sharpening, verifier gate, memory, and rollout budget can each be isolated or removed.

## 4. Methodology and Pseudocode

The pipeline has two phases. **Offline**, on the dev set, archive and memory are co-constructed together. **Online**, the archive is frozen and memory continues to grow if we configure it to.

The reason for the co-construction loop is concrete: prompts evolved without memory present will not have learned to use retrieved memory at inference time. So we alternate. Build memory using the current archive on hard dev queries, then evolve prompts with the current memory in context, then repeat. Hard-sample mining following P2O focuses prompt evolution on dev queries where the current archive fails, not on easy ones.

```python
# Offline: co-construction on dev set
for round in 1..R:
    # Build memory using current archive on hard dev queries
    for q in D_hard:  # low pass@K under current archive
        for prompt_i in current_archive:
            mem_i = retrieve(prompt_i, q)
            y ~ MixedAlphaSample(base_model | prompt_i, mem_i, q)
            if Verifier(q, y):
                admit distill(y) to memory
    # Evolve prompts against current memory
    propose new prompts via GEPA reflection
    score each prompt on dev set WITH current memory in context
    update archive via Pareto/MAP-Elites over
        {accuracy_with_memory, descriptor_novelty, cost}
freeze archive

# Online: inference for query q
for prompt_i in archive:
    mem_i = retrieve(prompt_i, q)
    for j in 1..n_i:
        y_{i,j} ~ MixedAlphaSample(base_model | prompt_i, mem_i, q)
        c_{i,j} = Verifier(q, y_{i,j})
return argmax_{i,j} y_{i,j} by c_{i,j}

# After inference (if online memory enabled)
if any y_{i,j} passes independent verifier check:
    admit distill(y_{i,j}) to memory with reliability prior
update reliability posteriors on retrieved memory entries
```

The verifier is task-specific:

- **Math:** symbolic or numeric equivalence.
- **Code:** unit-test execution.
- **Science-MC:** answer-key oracle for offline analysis only; not deployable as inference-time verifier.

Selector and offline-track rules:

- For verifier-covered tracks (math, code), the selector is `argmax verifier score` with a pre-registered tie-breaker.
- For offline-only tracks (GPQA-Diamond), POLARIS must use a pre-registered non-oracle selector (for example, majority vote over extracted multiple-choice answers with ties broken by normalized model likelihood or SPS score). Answer keys may evaluate the selected answer and oracle coverage; they may not choose the answer unless the result is explicitly labeled `oracle selection`.

Memory specifics (verifier-gated admission, Beta-Bernoulli reliability, retrieval bounds, distillation, leakage screening) are detailed in §8.

## 5. Experimental Setup

We inherit the benchmark and model setup from the sampling-baseline literature [R2, R3] so their RL reference numbers transfer directly.

### 5.1 Three task families

| Track | Benchmark | Evaluator | Inference-time verifier? | Regime expectation |
|---|---|---|---|---|
| Math | MATH500 [R11] | symbolic / numeric equivalence | yes | Diminished — strongest POLARIS signal |
| Code | HumanEval+ [R13, R14] | unit-test execution | yes | often Sustained — expect underperformance vs RL |
| Science MC | GPQA-Diamond [R12] | answer-key oracle | no, offline only | Diminished or Plateau |

Split policy follows the distinction used in the cited literature. Pure
sampling baselines report full benchmark scores directly [R2, R3]. POLARIS
learns prompts, archive selection, and memory, so optimizer feedback must come
from non-final pools:

- **Math optimizer-dev:** MATH train-style rows outside MATH500; final report:
  full MATH500 `0:500`.
- **Code optimizer-dev:** MBPP+/APPS/LiveCodeBench-style rows outside
  HumanEval+; final report: full HumanEval+ `0:164`.
- **Science optimizer-dev:** GPQA non-Diamond rows only if science prompt or
  memory development is needed; final report: full GPQA-Diamond `0:198`.

ProRL's regime analysis [R19] predicts where POLARIS should and should not work:

- **Math** is heavily pretrained (Diminished regime), so RL there is mostly sharpening and POLARIS should be competitive.
- **Code** is often Sustained: RL genuinely expands boundaries, so we expect POLARIS to underperform.
- **GPQA** is probably Diminished or Plateau.

The bifurcation across regimes is itself a result worth reporting cleanly, even if (especially if) POLARIS does not win uniformly.

### 5.2 Models

Direct replication of the sampling-baseline literature uses:

- `Qwen/Qwen2.5-7B` (RWS HumanEval direct match)
- `Qwen/Qwen2.5-Math-7B` (RWS MATH500 direct match)
- `deepseek-ai/deepseek-math-7b-base` (SPS direct match)
- A larger-model run (`meta-llama/Llama-3.1-70B` [R16] or similar) if compute allows.

`Qwen/Qwen3-8B-Base` [R15] is retained as a generalization checkpoint but is not the direct replication model for the cited sampling tables.

### 5.3 Baselines

Minimum baseline suite:

1. Greedy decoding (frozen base model).
2. Best-of-N (temperature sampling).
3. Single-prompt scalable power sampling.
4. Single best archive prompt (alone, with power sampling).
5. Dynamic Cheatsheet [R6] — LLM self-judged memory.
6. ACE [R7] — context evolution.
7. GEPA [R5] — prompt evolution without sharpening/memory at inference.
8. P2O [R18] where applicable — note P2O distills to weights; matched-budget comparison only at deployment phase.
9. Published GRPO reference numbers where the model-benchmark pairing matches [R2, R3].

A new GRPO reproduction is required only when published tables do not cover the exact model-benchmark combination.

### 5.4 Primary evaluation axis: total rollouts

Following GEPA [R5], the primary axis is **total rollouts at matched accuracy**:

```text
RolloutTotal(POLARIS, N) = ArchiveConstructionRollouts + N * InferenceRolloutsPerQuery
RolloutTotal(GRPO,    N) = TrainingRollouts + N
```

The headline number is the **break-even N**, the deployment volume at which POLARIS's amortized rollouts beat GRPO's.

Token count, dollar cost, and latency are secondary diagnostics. They are reported in `costs.json` for every run but they do not arbitrate the primary claim.

Archive-construction rollouts must count. GEPA-style search runs around 700 rollouts in their setup; POLARIS will probably want a larger archive, 2K-5K. GRPO training is ~24K. The break-even N calculation absorbs this, but the archive-construction line cannot be hidden.

If RL training rollouts or cost are unavailable for a baseline, label the result as an accuracy comparison against a published RL reference, not a full rollout comparison.

## 6. Ablations

Ablations run in layers, each isolating a different knob in the inference-time manifold. Every ablation reports rollouts at matched accuracy in addition to accuracy at matched rollouts.

### 6.1 Archive size

`k in {1, 4, 8, 16, 32}` at matched compute. Tests whether prompt-induced support expansion saturates, and where.

### 6.2 Memory composition

`{off, raw traces, distilled strategies, cross-task transfer}`, with `pass@K` plotted as a function of deployment volume `N`. Tests whether memory amortizes across deployment.

### 6.3 Diversity preservation

`{fixed alpha, mixed alpha, decaying alpha}`. Tests whether entropy preservation matters more on Sustained tasks than on Diminished tasks, as we expect.

### 6.4 Joint optimization

Best combination from §6.1-§6.3 vs. published GRPO and P2O.

### 6.5 Factorial interaction

A `2 x 2 x 2` factorial design over `{archive, sharpening, memory}` tests whether the three mechanisms compose multiplicatively or are redundant. The memory factor is also swept across deployment volume `N`, since amortization is the whole point of the mechanism.

Interaction test:

1. Fit a factorial regression on per-query correctness with binary factors `coverage`, `sharpening`, and `memory`.
2. Report the three-way interaction term on a logit-transformed scale.
3. Use bootstrap confidence intervals clustered by query.

Call the mechanisms **complementary** only if the relevant interaction terms are positive under the pre-registered scale. Do not call the effect "multiplicative" unless the logit-scale three-way interaction supports that claim.

### 6.6 Descriptor ablation

Replace trace-derived descriptors with surface, random, or validation-only descriptors under the same archive size and sample budget. Pre-check via dev-set regression of success on descriptors (cheap precheck described in §7.1).

### 6.7 Verifier-gating ablation

Replace external verification with LLM-judged memory curation. Measure false memory admissions via the post-hoc stronger-verifier audit (§8).

## 7. Risks and Caveats

### 7.1 Descriptor signal might not exist

The archive is indexed by trace features (reasoning pattern, depth, branching, verification density), and the story relies on those features actually correlating with whether a trajectory is correct. If they do not, the archive will be diverse in irrelevant ways and saturation in the size ablation will kick in early.

**Pre-check before committing to the full experiment:** a dev-set regression of success on descriptors. If descriptors carry no signal, the descriptor-alignment contribution is dead and we either pivot to a different archive criterion or scale POLARIS down to a sharpening-plus-memory paper.

### 7.2 Verifier quality is load-bearing in a one-sided way

False positives are the dangerous failure mode: they admit bad strategies that poison memory and propagate to later queries. False negatives just shrink memory, which is recoverable.

Risk varies a lot by benchmark:

- **Math:** near-zero under symbolic equivalence.
- **Code:** nonzero — which is why we use HumanEval+ over HumanEval [R14].
- **GPQA:** zero on the answer-key oracle, but the oracle itself is not deployable.

False-positive memory admission is estimated by a post-hoc audit: sample admitted entries and LLM-curated baseline entries, replay them on held-out verifier-covered queries, and count entries whose source solution or retrieval-time use fails a stronger verifier or manual adjudication. This estimates false positives; it is not treated as perfect ground truth.

### 7.3 Archive cost has to count

GEPA-style search runs around 700 rollouts in their setup [R5]; POLARIS will probably want 2K-5K. GRPO training is ~24K. The break-even N calculation absorbs this, but the archive-construction line must be reported explicitly in `costs.json` for every run.

### 7.4 POLARIS targets sharpening, not boundary expansion

This is the biggest caveat. ProRL [R19] shows that on Sustained-regime tasks, prolonged RL with entropy preservation accesses trajectories the base model genuinely cannot produce (`pass@128 = 0 -> ProRL pass > 0`). Power sampling preserves support, which means it cannot reach those trajectories on its own. Memory partially closes the gap by injecting new context, but only for trajectories that get seeded into memory in the first place.

We expect to be strongest on the Diminished regime and weakest on the Sustained regime. This is not a failure of the proposal; it is the answer to the central question.

### 7.5 The composition might be redundant

Any one of `{archive, sharpening, memory}` might capture most of the inference-time gain alone, and regime probably determines which mechanism does the work where. The factorial ablation (§6.5) is designed to catch this, and the regime-conditioned analysis is what makes the result interpretable rather than just "matches or doesn't."

## 8. Verifier-Gated Memory (detail)

POLARIS stores transferable strategies, not raw solution traces.

Memory has two separate processes:

1. **Admission**: a new memory entry is admitted only after an independent verifier check succeeds on the query where the strategy was discovered.
2. **Reliability accumulation**: reliability updates later when a stored memory entry is retrieved for a new query and the resulting candidate is accepted or rejected by the verifier.

This separation is necessary because a Beta reliability model must observe both successes and failures.

If `z in {0, 1}` is the verifier outcome after a retrieved memory entry is used on a later query, a memory entry with posterior `Beta(a, b)` updates to `Beta(a + z, b + 1 - z)`. The posterior mean `a / (a + b)` is used for memory ranking. Low-reliability entries are pruned only under a pre-registered threshold.

Memory retrieval protocol:

1. Filter memory entries to the same benchmark track and compatible verifier type.
2. Remove entries whose posterior reliability lower confidence bound is below the pre-registered pruning threshold.
3. Rank remaining entries by a pre-registered score combining descriptor similarity, embedding similarity (if used), posterior reliability, and token cost.
4. Retrieve at most `max_retrieved_memory_entries` entries and at most `max_retrieved_memory_tokens` tokens.
5. Log all retrieved and eligible-but-not-retrieved memory ids.

Memory pool growth is bounded. Each archive entry may hold at most `max_memory_entries_per_archive` memory entries. Pruning happens after admission using the pre-registered rule, so memory cannot become an unbounded hidden ensemble.

Strategy distillation protocol:

1. Prefer deterministic extraction templates that remove problem-specific constants and preserve only transferable tactics.
2. If an LLM distiller is used, log model, prompt, temperature, schema, and cost. Count the call in `MemoryCost`.
3. The distiller cannot see held-out labels or answer keys.
4. The resulting strategy must pass a leakage screen that rejects raw final answers, copied full solutions, benchmark ids, or query-specific constants unless the task type makes the constant itself the reusable object.

`independent_check` means a second verifier path not identical to `verifier.score`. Examples: code rerun under an EvalPlus or hidden-test subset not used for ranking; math reparse plus symbolic/numeric equivalence through a separate extractor; GPQA has no exact independent inference-time check, so online memory admission is disabled there unless a non-oracle verifier is explicitly defined and labeled.

Memory experiments must report: admission count, estimated false-positive admissions, estimated false-negative rejections, retrieval-time success rate, reliability posterior summaries, and downstream performance after memory growth.

Default constants, unless a run pre-registers alternatives:

```text
max_memory_entries_per_archive = 256
max_retrieved_memory_entries   = 3
max_retrieved_memory_tokens    = 512
memory_reliability_prior       = Beta(1, 1)
memory_prune_rule              = reliability_lower_bound_then_descriptor_coverage_then_cost
```

## 9. Trace-Aligned Descriptors

POLARIS's primary novelty claim is descriptor alignment.

**Hypothesis:** a prompt archive indexed by trace-derived reasoning descriptors gives better fixed-budget coverage than an archive indexed by validation score alone, random diversity, or prompt-surface features.

The descriptor is computed from generated traces rather than prompt text alone.

For each trace `tau`, the descriptor extractor estimates five features:

1. **Pattern label**: one of `direct computation`, `algebraic transformation`, `case analysis`, `contradiction`, `backward verification`, `induction`, `search/enumeration`, `mixed/other`.
2. **Step count**: number of parsed inferential, computational, or tool-use steps.
3. **Branch count**: number of explicit cases, alternatives, or explored paths.
4. **Verification density**: fraction of steps containing an explicit check, substitution, boundary test, unit test, or consistency check.
5. **Symbol / object diversity**: normalized diversity of mathematical symbols, program variables, entities, or intermediate objects manipulated in the trace.

For each prompt `p_i`, the system samples `K` development traces and aggregates their descriptor statistics.

The archive keeps prompts that are useful, behaviorally nonredundant, and not dominated by cheaper prompts with similar validation score and descriptor coverage.

Descriptor reliability is measured before the main experiment because the descriptor extractor is a measurement instrument.

Pre-registered descriptor audit:

1. Sample 200 traces per benchmark track.
2. Label each trace with two independent judges.
3. Report categorical agreement for pattern label with Cohen's kappa or an equivalent multi-rater statistic.
4. Report bootstrap confidence intervals for numeric descriptor agreement.

The descriptor extractor must be pre-registered as one of:

1. heuristic parser with versioned code path;
2. LLM judge with fixed model, prompt, temperature, output JSON schema, and cost accounting;
3. hybrid parser plus LLM fallback with explicit fallback triggers.

The extractor cannot see held-out labels, answer keys, or verifier outcomes except for source-query pass/fail status already logged after verification.

The descriptor ablation (§6.6) tests utility. The inter-judge reliability audit tests measurement quality. If trace-derived descriptors do not outperform surface, random, or validation-only descriptors under the same archive size and sample budget, descriptor alignment is not the contribution.

## 10. Archive Construction

Archive construction is offline.

A GEPA-style reflective prompt optimizer [R5], MAP-Elites-style search [R10], or another prompt-mutation procedure proposes candidate prompts on a development set. Each candidate prompt is scored by:

1. validation performance with current memory in context (co-construction loop);
2. descriptor coverage;
3. average inference rollouts and tokens;
4. verifier compatibility.

Initial archive-size sweep: `k in {4, 8, 16, 32}`.

The selection rule is Pareto or MAP-Elites-style over `{accuracy_with_memory, descriptor_novelty, cost}`.

The archive is frozen before held-out evaluation. Online memory may continue to grow at evaluation time only in experiments explicitly labeled as `online verified-memory`.

## 11. Success Criteria

- **Primary success:** higher task accuracy than the strongest inference-time baseline at the same rollout budget.
- **Stronger success:** matching or exceeding a published GRPO-style reference baseline at matched or lower total rollouts on at least one directly comparable model-benchmark setting.
- **Descriptor-alignment success:** trace-derived descriptors outperform random, surface, or validation-only archives under matched archive size and rollout budget.
- **Verifier-gated-memory success:** memory growth improves downstream performance while reducing false-positive memory admissions relative to LLM-judged curation.
- **Regime-bifurcation success:** the regime-conditioned story holds — POLARIS wins or competes in Diminished regimes (math, possibly GPQA) and underperforms in Sustained regimes (code). Reporting the bifurcation cleanly is itself a publishable result.
- **Break-even success:** POLARIS's amortized rollouts beat GRPO's at deployment volume `N <= N*`, where `N*` is pre-registered.

## 12. Falsification Rules

- If POLARIS fails to beat the strongest inference-time baseline on any track, the composition claim fails on that track.
- If POLARIS beats inference-time baselines but not the GRPO-style reference on its strongest regime (math), the result becomes a training-free inference improvement rather than an RL-substitution result.
- If trace descriptors do not outperform surface, random, or validation-only descriptors, descriptor alignment is not the contribution.
- If verifier-gated memory does not reduce false admissions or improve downstream accuracy, memory is removed from the core method.
- If performance improves only by spending substantially more rollouts than GRPO training, the result must be framed as a rollout-accuracy tradeoff rather than an efficient substitute for post-training.

## 13. Active Execution Gate

The previous fixed-alpha HumanEval+ Phase 0-2 gate remains useful only as a carryover replication baseline for Reasoning with Sampling on `Qwen/Qwen2.5-7B`. C0c is complete (HumanEval 0.5549 vs RWS 0.622, within tolerance).

Current live work:

- **POLARIS publishable-readiness infrastructure**: full-method condition,
  split-aware dataset locks, generic production plan, artifact audit, trajectory
  cache replay, and bounded backend smokes.
- **Current split contract**: optimizer-dev rows are external to the final
  benchmark where possible; final reporting uses full MATH500, full HumanEval+,
  and full GPQA-Diamond. GPQA remains gated on accepted HF access or official
  local cache.
- **Next runnable evidence**: one cached real row per track/condition, then
  small real slices, then descriptor audit and real archive-memory construction.
  Any paid launch still requires a fresh explicit command and cost cap.
- **ProRL recovery audit amendment**: the standalone ProRL/BroRL Recoverable
  Fraction audit now uses a multi-resource live scheduler. FarmShare L40S is
  the free continuation path, Mithril/Flow A100 is the capped weekend
  accelerator, Modal is Phase 3/debug burst capacity, CloudRift is fallback,
  and xAI provides GEPA reflection. The checked-in resource profile and each
  launch artifact must record backend, cache, split, cost estimate, cost cap,
  and explicit per-launch authorization.

The next expensive experiment must be bounded, costed in rollouts (and dollars as diagnostic), and pre-registered before launch.

## 13.1 R5 Infrastructure Contract

R5 exists to make POLARIS cheap enough to run honestly. The goal is not a new scientific condition; it is the execution substrate for the pre-registered conditions.

Backend policy:
- **vLLM V0 is the optimized MCMC candidate** because its per-request logits processor can observe full-vocabulary logits and force chosen target tokens. Current accepted configuration is `dtype=float32`, `model_impl=transformers`, `VLLM_USE_V1=0`, and `enable_prefix_caching=True`.
- The accepted vLLM implementation must batch candidate chains inside each MCMC block/proposal step and the MATH500 runner must batch cache-miss candidates per problem; scalar per-candidate MCMC is not cost-ready.
- The GPU-accepted vLLM MCMC path uses native vLLM sampling for draft/proposal tokens, then a batched forced-token scoring pass for MH logprobs. The fused "sample inside logits processor" path is opt-in only and not accepted: Modal H100 diverged between forced and emitted tokens at token 105 in a 128-token batched decode.
- vLLM native `logprobs` / `prompt_logprobs` is not an accepted shortcut for MH scoring: the bounded Modal probe failed inside vLLM V0's sampler assertion before producing usable per-token scores.
- **HF/RWS remains the correctness oracle** for MCMC scoring because it exposes direct full-vocabulary logits through `model(input_ids).logits`.
- **SGLang remains useful for shared-prefix generation only, not accepted MCMC scoring**. Its forced-token scorer failed HF parity by orders of magnitude in bounded Modal tests, so it cannot drive MH ratios until that is fixed.
- **Rejected vLLM sketch:** source check invalidated `SamplingParams(max_tokens=0)`; scoring uses forced decode instead.

Cost-control invariants:
- Local infrastructure readiness must be proven before paid backend work: `scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp` exercises the CPU experiment loop, mandated artifacts, trajectory cache, HumanEval+/GPQA loaders and verifiers, archive/memory/diversity/factorial/descriptor/verifier-gating/break-even scenario drivers, and records no deferred rows unless a real external blocker is proven.
- Paid run entrypoints, including Modal, CloudRift, and Mithril/Flow GPU smokes, must fail closed unless a preflight specifies artifact directory, trajectory cache, split, seed, model, backend, estimated cost, cost cap, and explicit user authorization.
- Every generated trajectory is cacheable by `(model_id, track, problem_id, prompt_id, sample_idx, alpha, seed)`.
- Ablations must replay cached trajectories whenever the `(model, track, prompt, alpha, seed)` cell already exists.
- Modal may be used for short R5 smokes and Phase 3/debug bursts. Mithril/Flow bulk generation, Flow A100 acceleration, and MATH500 Phase 10 require a fresh explicit user command.
- No serving backend can drive MCMC until `score_segments` parity is below `1e-3` absolute difference against the HF oracle on a bounded Modal smoke. vLLM V0 now passes this gate in the bounded scorer smoke; scale-up still requires a fresh user command.
- No scale-up until `scripts/modal_vllm_app.py::smoke_vllm_batched_power_path` passes and the observed batched wall time projects under the R5 cost gate. The correctness smoke has passed; the exact Phase 10 cost projection still must clear before any real run.

## 14. Drift-Lock Protocol

Cross-agent failure modes and hard blocks:

1. **Protocol mismatch:** any run command that would execute against a phase contract different from `TODO.md` or `runs/progress.md` fails validation.
2. **Hyperparameter drift:** changing locked values (`alpha`, `K_t`, `M_t`, `block_num`, `num workers`, `seed`, `B`, `k`) without protocol amendment is disallowed.
3. **Evidence mismatch:** a phase is not complete until artifacts and decisions are mirrored in `runs/progress.md`.
4. **Scope creep:** no expensive experiment may start until its alpha schedule, memory setting, benchmark regime, split, rollout budget, and stop rule are written in `TODO.md` and `runs/progress.md`.
5. **Result fabrication risk:** every reported number must link to an artifact path (`jsonl`, `log`, or `manifest`) before being treated as final.

Compliance checklist before any phase transition:

- `bash scripts/check_protocol_sync.sh` passes.
- TODO status is `ready to transition`.
- Progress has a full section for the completed phase with criteria outcomes and blockers.
- No unresolved mismatched statements remain across the three contract files.

If any check fails, execution halts and status becomes `blocked` with a root-cause entry in `runs/progress.md`.

## 15. Pre-flight Before GPU

Before every paid GPU launch (CloudRift, Modal, Mithril, Flow, Colab paid tier):

1. Re-read the source paper's Table 1 / experimental section for **this benchmark**.
2. Verify base model, hyperparameters, dataset split, and rollout/cost budget against the paper.
3. Never inherit a model or config from a prior run on a different benchmark — different benchmarks in the same paper often use different bases (e.g., RWS Table 1: HumanEval = `Qwen/Qwen2.5-7B`, MATH500 = `Qwen/Qwen2.5-Math-7B`).
4. Show the verified spec to the user and wait for an explicit per-launch `go`. Prior frustration is not standing consent.

## 16. Agent Decomposition for Execution

Each agent owns a named responsibility and edits only designated surfaces:

1. **Protocol steward** — only this agent updates `PROPOSAL.md`, `TODO.md`, `runs/progress.md`; owns the active gate and amendment log.
2. **Environment/setup agent** — provisioning, dependency install order, GPU image build, run-volume layout; reports health checks only.
3. **Replication/baseline agent** — owns RWS/SPS direct-match scripts and preflight checks; cannot change locked hyperparameters.
4. **Archive agent** — owns GEPA-style prompt-mutation wrapper, MAP-Elites grid, and archive outputs; enforces `k`, dev split, and pruning checks.
5. **Inference agent** — owns condition runners, candidate JSONL schema, and selector logic.
6. **Memory agent** — owns admission gate, Beta-Bernoulli reliability updates, retrieval bounds, distillation/leakage screen.
7. **Scoring + stats agent** — owns verifier execution, bootstrap CI, McNemar, factorial regression, and break-even-N calculation.

Shared rule: no agent writes to another agent's surface except through explicit handoff notes in `runs/progress.md` and a reviewed protocol amendment in `PROPOSAL.md`.

## 17. Claims Removed or Corrected

1. Acronym expanded to **POLARIS: Prompt-Organized Library of Archived Reasoning and Inference Strategies** (v3.0).
2. The original "substitutes for RL post-training" framing is replaced by a benchmark-and-regime-specific, falsifiable hypothesis: recover what is recoverable, fail cleanly where ProRL boundary expansion is required.
3. The primary evaluation axis is **total rollouts at matched accuracy** (break-even N), not dollar cost. Cost is a diagnostic.
4. The co-construction loop (offline) and online memory growth are the methodology, not an extension. Memory disabled in v1 is a deliberate isolation experiment, not the design intent.
5. Mixed-alpha sampling is a core mechanism, not an ablation knob.
6. The Coverage Principle is theoretical motivation; empirical archive oracle coverage is reported as a separate Pass@B-style metric.
7. CycleQD is parameter-level QD model merging with task-performance behavioral characteristics, not a prompt-surface descriptor precedent.
8. The memory algorithm separates verifier-gated admission from retrieval-time reliability updates that observe both successes and failures.
9. The RWS carryover replication uses locked `alpha = 4`; POLARIS treats the alpha schedule as a pre-registered design variable.
10. Descriptor validation commits to an inter-judge reliability audit.
11. The interaction claim uses a pre-registered factorial regression on a logit-transformed scale rather than an informal "multiplicative" claim.
12. GPQA answer keys are restricted to offline evaluation or explicitly labeled oracle selection.
13. Qwen3-8B-Base is retained as a generalization model but removed as the direct replication model for the cited sampling tables.
14. Three task families are explicitly committed: MATH500 (Diminished), HumanEval+ (Sustained, expected to underperform), GPQA-Diamond (Diminished/Plateau, offline only).
15. Three replication models are committed: Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B; larger-model generalization optional.

## 18. Reference Audit

Verification scope: title, authors, date/version, and proposal-relevant support were checked against primary arXiv pages/PDF text where available. Hugging Face model names were checked against model cards.

| ID | Reference | Verified support | Proposal use | Constraint |
|---|---|---|---|---|
| R1 | Zhao et al., [Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining](https://arxiv.org/abs/2504.07912), arXiv:2504.07912 | Studies RL fine-tuning from scratch across data mixtures and reports convergence toward dominant output distributions that amplify pretraining patterns. | Motivation for reweighting and amplification. | Does not prove all RL post-training is only distribution sharpening. |
| R2 | Karan and Du, [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901), arXiv:2510.14901 | Supports training-free MCMC-style sampling from sharpened distributions and reports comparisons to GRPO on MATH500, HumanEval, and GPQA. | Sampling baseline and published RL reference source. | Do not attribute scalable token-level approximation to this paper. |
| R3 | Ji et al., [Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening](https://arxiv.org/abs/2601.21590), arXiv:2601.21590 | Supports autoregressive/token-level approximation to power sampling, MATH500/HumanEval/GPQA evaluation, GRPO reference rows, and `alpha = 4`, `K_t = M_t = 8`. | Main sharpening mechanism and direct replication target. | Treat the scaling factor as an approximation and lock hyperparameters before held-out evaluation. |
| R4 | Chen, Huang et al., [The Coverage Principle: How Pre-Training Enables Post-Training](https://arxiv.org/abs/2510.15020), arXiv:2510.15020 | Supports coverage as a theoretical lens for Best-of-N, test-time scaling, and post-training success. | Conceptual basis for archive coverage. | Do not equate formal `Cov_N` with empirical archive oracle coverage. |
| R5 | Agrawal et al., [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457), arXiv:2507.19457 | Supports reflective prompt evolution, natural-language reflection, Pareto-frontier candidate management, and ~35x fewer rollouts than RL on its benchmarks. | Archive-construction machinery; rollout-axis precedent. | Do not claim GEPA discards frontier information unless comparing to a specific deployment protocol that does so. |
| R6 | Suzgun et al., [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952), arXiv:2504.07952 | Supports persistent evolving memory without explicit ground-truth labels or human feedback. | Memory baseline. | Memory corruption is a measured risk, not an assumed failure. |
| R7 | Zhang et al., [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618), arXiv:2510.04618 | Supports evolving playbooks through generation, reflection, curation, and execution feedback. | Context-evolution baseline. | Do not reduce ACE to simple LLM-judged curation. |
| R8 | Kuroki et al., [Agent Skill Acquisition for Large Language Models via CycleQD](https://arxiv.org/abs/2410.14735), arXiv:2410.14735 | Supports QD-style LLM agent skill acquisition through model archives, model/task-vector merging, SVD-based mutation, and task-performance behavioral characteristics. | Related QD work. | CycleQD is parameter-level model merging, not a prompt-level archive method. |
| R9 | Chen et al., [Learning to Self-Evolve](https://arxiv.org/abs/2603.18620), arXiv:2603.18620 | Supports RL training of a 4B self-evolving policy for test-time context improvement on BIRD and MMLU-Redux. | Trained context-evolution comparator. | Not a direct MATH500/HumanEval/GPQA baseline unless reproduced there. |
| R10 | Mouret and Clune, [Illuminating Search Spaces by Mapping Elites](https://arxiv.org/abs/1504.04909), arXiv:1504.04909 | Supports MAP-Elites and the quality-diversity archive framing. | Methodological ancestry. | Descriptor choice must be justified empirically. |
| R11 | Hendrycks et al., [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874), arXiv:2103.03874; Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), arXiv:2305.20050 | Supports MATH provenance and the 500-problem MATH test subset used in process-supervision evaluations. | Math benchmark citation. | Report which problem types are covered by symbolic, numeric, or exact-string verification. |
| R12 | Rein et al., [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022), arXiv:2311.12022 | Supports GPQA as an expert-written multiple-choice science benchmark; sampling papers use GPQA-Diamond as the 198-question high-quality subset. | Science-QA evaluation. | Answer keys are offline evaluators, not deployable inference verifiers. |
| R13 | Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374), arXiv:2107.03374 | Supports HumanEval and pass@k-style code-generation evaluation. | Code benchmark citation. | Unit tests measure functional correctness but are not formal proofs. |
| R14 | Liu et al., [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210), arXiv:2305.01210 | Supports EvalPlus and HumanEval+ as stronger test suites that catch additional wrong generated code. | Preferred code verifier. | Stronger tests reduce but do not eliminate false accepts. |
| R15 | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) model card | Confirms the Qwen3-8B-Base checkpoint exists and is an 8B-class base checkpoint. | Generalization model. | Not the direct replication model for the cited sampling tables. |
| R16 | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) model card | Confirms the official pretrained Llama 3.1 70B checkpoint name. | Larger-model generalization. | Use official checkpoint names and account for gated access. |
| R17 | Novikov et al., [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://arxiv.org/abs/2506.13131), arXiv:2506.13131 | Supports LLM-driven evolutionary search over candidate programs/algorithms with automated evaluation. | Supporting prior for evaluated evolution and archive-style search. | Do not overclaim it as a prompt-archive or verifier-gated-memory precedent. |
| R18 | Lu et al., [P2O: Reinforcement Learning with Verifiable Rewards from Probabilistic Prompt Optimization](https://arxiv.org/abs/2603.21877), arXiv:2603.21877 | Frames prompt `z` as a latent variable, optimizes prompt-conditioned trajectories, and distills resulting behavior into weights. | Closest conceptual predecessor for prompt-latent support expansion and hard-sample mining. | POLARIS keeps gains at inference time and memory rather than distilling them into model weights. |
| R19 | Liu et al., [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864), arXiv:2505.24864 | Supports a regime lens where prolonged RL can preserve entropy and solve some tasks that remain unreachable by base-model sampling. | Main counterpoint and central question target. | Mixed-alpha sampling is only an inference analog to entropy preservation, not a replacement for RL training. |

## 19. Advisor Recommendation

Proceed with POLARIS as a training-free inference-time composition study oriented around the central question: which ProRL-style gains can be recovered with pure inference?

The proposal is strongest if it proves four claims under matched rollouts:

1. archive-wide sampling beats the single best prompt;
2. scalable power sampling with diversity preservation improves local yield over fixed-alpha;
3. verifier-gated memory reduces false admissions vs LLM-judged curation;
4. trace-derived descriptors outperform random or surface descriptors.

The proposal is weakest if it relies on oracle answer keys, unbounded archive growth, unverifiable memory, post-hoc hyperparameter tuning, or unsupported claims that post-training never creates new capabilities.

Expected publishable outcomes:

1. a new inference-time method; or
2. a clear regime-conditioned boundary result showing when inference-time composition can and cannot replace post-training.
