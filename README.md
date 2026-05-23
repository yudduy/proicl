# ProICL Experimetn command

Run the experiment on a CUDA host with `bash scripts/run_experiment.sh`;

it installs deps, calibrates SPS/vLLM, runs the held-out Reasoning Gym slice, audits artifacts, and packages results.
The result bundle is written automatically to `runs/experiment/<run-id>/results_bundle.tar.gz`.
Default run time estimate: 
1x A100-80GB ~16-27h, 
2x A100 ~9-15h, 
4x A100 ~6-8h, 
1x H100 ~9-16h, 
4x H100 ~4-6h; 
for a quick smoke use `EVAL_END=22 ROLLOUT_BUDGET=2 bash scripts/run_experiment.sh`.
