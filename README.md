# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment.sh auto`

Use `bash scripts/run_experiment.sh h100`, `bash scripts/run_experiment.sh a100`, or `bash scripts/run_experiment.sh l40` to force a profile. The profile wrappers remain as compatibility aliases only. The script auto-detects assigned GPUs, caps safe concurrency, selects a compatible vLLM dtype, requires Python 3.11/3.12 for the pinned vLLM stack, logs W&B when `WANDB_API_KEY` is set, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Resume the latest matching run with `bash scripts/run_experiment.sh auto --resume latest --progress-interval 60`.
Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
