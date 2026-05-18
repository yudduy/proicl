# POLARIS

POLARIS studies which RL-style reasoning gains can be recovered from a frozen base model through inference-time composition: prompt-archive spread, diversity-preserving power sampling, and verifier-gated memory.

Current protocol: `POLARIS-v3.1`. Treat `PROPOSAL.md`, `TODO.md`, and `runs/progress.md` as the scientific and operational contract.

## Repository Boundary

Tracked source should include:

- protocol docs: `PROPOSAL.md`, `TODO.md`, `AGENTS.md`, `runs/progress.md`
- package code under `src/polaris/`
- launch and analysis scripts under `scripts/`
- tests under `tests/`
- small configs, locks, and fixture archives under `configs/` and `data/`
- vendored runtime code under `src/polaris/vendored/`

Generated run artifacts are intentionally ignored under `runs/` except `runs/progress.md`. Reference clones under `upstream/` are also ignored and documented in `upstream/README.md`.

## Setup

```bash
python -m venv .venv-eval
./.venv-eval/bin/python -m pip install -U pip
./.venv-eval/bin/python -m pip install -e ".[code,dc,gepa_reflection]"
```

Copy `.env.example` to `.env` for local secrets. Do not commit `.env`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python -m pytest -q tests/unit/ tests/smoke/
bash scripts/check_protocol_sync.sh
PYTHONDONTWRITEBYTECODE=1 ./.venv-eval/bin/python scripts/smoke_polaris_readiness.py --out runs/readiness_smoke.tmp
```

Paid or scale runs require the explicit launch fields and user authorization described in `AGENTS.md`.
