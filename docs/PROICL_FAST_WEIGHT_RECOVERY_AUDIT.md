# ProICL Fast-Weight Recovery Audit

Status: protocol draft for the ProICL session. This supersedes older POLARIS
framing for this experiment only.

## Question

Which part of ProRL's slow-weight gain can be reproduced by frozen-weight fast
adaptation?

The audit is not a claim that inference replaces RL. It decomposes the ProRL
gain into:

- exploitation of the base conditional;
- context-conditional discovery;
- their composition;
- verifier-gated memory accumulation;
- residual slow-weight gain.

## Mechanism

RL with verifiable rewards couples three operations:

1. rollout discovery from `pi_theta(y | x, z)`;
2. advantage-weighted parameter update, changing slow weights `theta`;
3. diversity preservation through entropy, KL, and exploration machinery.

ProICL keeps `theta` frozen and tests the closest fast-weight analogues:

- GEPA searches over contexts `z`, using natural-language reflection and Pareto
  diversity. It searches `{pi_theta(y | x, z) : z in Z}`.
- MCMC power sampling samples from a sharpened `pi_theta(y | x, z)^alpha`.
  It exploits a conditional; it does not discover a new conditional.
- Verified memory stores external, verifier-admitted strategies and retrieves
  them as context. It is an external curriculum, not a weight update.

The critical asymmetry: GEPA plus MCMC is bounded by the effective support of
the prompted frozen model. ProRL can move slow weights and change the base
policy for future prompts. The experiment measures how large that asymmetry is.

## Primary Conditions

All frozen-base conditions use `deepseek-r1-distill-qwen-1.5b`.

| ProICL condition | Runtime condition | Interpretation |
| --- | --- | --- |
| `mcmc_only` | `single_prompt_power` | Direct prompt + power sampling; pure exploitation. |
| `gepa_only` | `gepa_only` | Cross-task evolved archive, no power sampling; fast-weight discovery. |
| `gepa_mcmc` | `proicl_gepa_mcmc` | Cross-task archive + fixed-alpha MCMC; core hypothesis. |
| `gepa_mcmc_memory` | `proicl_gepa_mcmc_memory` | Archive + MCMC + verifier-gated memory. |
| `prorl_v2_greedy` | `greedy` on `nemotron-prorl-v2@main` | Slow-weight reference. |

Base accuracy `A_base` comes from base greedy/pass@1 on the same task panel.

## Tracks

Primary panel:

- MATH500;
- GPQA-Diamond, with answer keys used only as offline evaluators;
- Reasoning Gym `boxnet`;
- Reasoning Gym `graph_color`;
- Reasoning Gym `family_relationships`.

The Reasoning Gym archive is cross-task by default. Prompt evolution may use
optimizer-feedback seeds from the Reasoning Gym dev slice only. Held-out eval
seeds must not feed GEPA, memory admission, checkpoint choice, selector choice,
or prompt edits.

## Decomposition

Report:

```text
A_base
A_mcmc
A_gepa
A_gepa_mcmc
A_memory
A_prorl_v2
```

Compute:

```text
RF_mcmc      = (A_mcmc      - A_base) / (A_prorl_v2 - A_base)
RF_gepa      = (A_gepa      - A_base) / (A_prorl_v2 - A_base)
RF_gepa_mcmc = (A_gepa_mcmc - A_base) / (A_prorl_v2 - A_base)
RF_memory    = (A_memory    - A_base) / (A_prorl_v2 - A_base)

slow_weight_residual = A_prorl_v2 - A_gepa_mcmc
memory_gain          = A_memory   - A_gepa_mcmc
```

The OpenReview ProRL ablation suggests a rough calibration: curriculum exposure
accounts for a large share of Reasoning Gym gain, while RL adds an additional
reliability term. The `~32 / 55 ~= 58%` figure is an interpretive reference,
not a target and not a decision rule.

## Gates

- No RF claim before the exact Karan-Du MATH500 RWS gate passes.
- No ProRL or BroRL trajectories may seed GEPA or the RF numerator memory.
- GEPA construction rollouts are charged separately from inference rollouts.
- GPQA oracle keys may evaluate selected answers but may not select them.
- Paid or bulk runs require explicit user authorization and a cost cap.

## Sources

- Tiwari et al., [Learning, Fast and Slow](https://arxiv.org/html/2605.12484v2).
- Karan and Du, [Reasoning with Sampling](https://arxiv.org/abs/2510.14901).
- Liu et al., [ProRL](https://arxiv.org/abs/2505.24864) and the
  [OpenReview thread](https://openreview.net/forum?id=YPsJha5HXQ).
- Agrawal et al., [GEPA](https://arxiv.org/abs/2507.19457).
