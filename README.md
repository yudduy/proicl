# ProICL Experiment

Run on the GPU machine:
`git clone git@github.com:yudduy/proicl.git && cd proicl && GPUS=0,1,2,3,4 bash scripts/run_experiment.sh`

Manual env setup: `pip install -r requirements-light.txt`, then run with `SKIP_INSTALL=1`.
The script auto-detects visible GPUs, shows live progress, and prints `Result bundle: /absolute/path/results_bundle.tar.gz`.
Fetch it from your local machine with:
`mkdir -p results && scp <cluster>:/absolute/path/results_bundle.tar.gz ./results/`
