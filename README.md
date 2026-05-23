# POLARIS SPS Recovery

Run the experiment on a CUDA host with `bash scripts/run_sps_recovery_experiment.sh`; it installs deps, calibrates SPS/vLLM, runs the held-out Reasoning Gym slice, audits artifacts, and packages results.
The result bundle is written automatically to `runs/mentor_sps/<run-id>/sps_results_bundle.tar.gz`.
Default run time estimate: 1x A100-80GB ~16-27h, 2x A100 ~9-15h, 4x A100 ~6-8h; for a quick smoke use `EVAL_END=22 ROLLOUT_BUDGET=2 bash scripts/run_sps_recovery_experiment.sh`.
