# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment.sh`

Use `bash scripts/run_experiment.sh h100`, `bash scripts/run_experiment.sh a100`, or `bash scripts/run_experiment.sh l40` only to force a profile. The script auto-detects assigned GPUs, resumes the latest incomplete matching run, emits active progress updates, caps safe concurrency, selects a compatible vLLM dtype, requires Python 3.11/3.12 for the pinned vLLM stack, logs W&B when `WANDB_API_KEY` is set, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Force a new run with `bash scripts/run_experiment.sh --fresh`.

On Sherlock, use the Python module only and let the repo create its own vLLM environment:
`ml python/3.12.1 && bash scripts/run_experiment.sh`
Do not load Sherlock's central `py-vllm`, `py-pytorch`, or `py-transformers` modules for this experiment; their advertised versions do not match the pinned vLLM stack.

Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
