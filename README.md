# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && bash scripts/run_experiment.sh`

The script auto-detects A100/H100 GPUs and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
