# ProICL Held-Out Experiment

Run: `bash scripts/run_experiment.sh`
It auto-detects visible A100/H100 GPUs, runs calibration + held-out eval, audits artifacts, and writes `runs/experiment/<run-id>/results_bundle.tar.gz`.
Use `GPUS=0,1` to limit GPUs, `GPU_PROFILE=a100|h100` to override detection, or `EVAL_END=22 ROLLOUT_BUDGET=2` for a quick smoke.
