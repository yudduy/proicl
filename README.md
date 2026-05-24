# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment_l40.sh`

Use `scripts/run_experiment_h100.sh` on H100 or `scripts/run_experiment_a100.sh` on A100. The script auto-detects assigned GPUs, caps safe concurrency, logs W&B when `WANDB_API_KEY` is set, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
