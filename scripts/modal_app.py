"""Modal entrypoint for POLARIS-MATH500 v1.

Smoke:
    modal run scripts/modal_app.py::smoke

Build archive (Phase 6 on GPU; emits archive.json into the polaris-runs Volume):
    modal run scripts/modal_app.py::build_archive

Run one condition × one seed:
    modal run scripts/modal_app.py::run_one --condition full_archive_fixed --seed 17

Sync run artifacts back to local:
    modal volume get polaris-runs / runs/<dest>/
"""

from __future__ import annotations

from pathlib import Path

import modal

POLARIS_ROOT = Path(__file__).resolve().parent.parent

_IGNORE = [
    ".venv*",
    "__pycache__",
    "*.pyc",
    "upstream",  # vendored copies are authoritative; upstream/ is 500MB of reference
    "runs/_archive",
    "runs/rws_official_full164_mcmc_ckpt",  # C0c artifacts, large
    ".git",
    ".pytest_cache",
    ".claude",  # local agent state (scheduled_tasks.lock writes can invalidate the image build mid-run)
    "tests",  # not needed for GPU runs
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.0",
        "accelerate>=1.0",
        "datasets",
        "huggingface-hub",
        "sympy",
        "pylatexenc",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "reasoning-gym",
    )
    .add_local_dir(str(POLARIS_ROOT), remote_path="/polaris", copy=True, ignore=_IGNORE)
)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.4.7-cu124", add_python="3.11")
    .pip_install(
        "datasets",
        "hf-transfer==0.1.8",
        "huggingface-hub",
        "sympy",
        "pylatexenc",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_XET": "1",
            "SGLANG_USE_MODELOPT_FOR_FP8": "1",
        }
    )
    .add_local_dir(str(POLARIS_ROOT), remote_path="/polaris", copy=True, ignore=_IGNORE)
)

app = modal.App("polaris-math500", image=image)
hf_cache = modal.Volume.from_name("polaris-hf-cache", create_if_missing=True)
sglang_cache = modal.Volume.from_name("polaris-sglang-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("polaris-runs", create_if_missing=True)


def _setup_paths() -> None:
    """Wire polaris on sys.path; point HF cache at the persistent Volume."""
    import os
    import sys

    if "/polaris/src" not in sys.path:
        sys.path.insert(0, "/polaris/src")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _require_modal_preflight(
    *,
    backend: str,
    estimated_dollar_cost: float | None,
    cost_cap_dollars: float | None,
    user_authorized_paid_run: bool,
    split: tuple[int, int] = (0, 1),
    seed: int = 17,
    artifact_dir: str = "/polaris-runs/modal-smoke",
    cache_path: str = "/polaris-runs/trajectories.sqlite",
) -> dict:
    from pathlib import Path as _Path

    from polaris.config import MODEL_ID
    from polaris.infra.preflight import PaidRunPreflight, validate_paid_run_preflight

    return validate_paid_run_preflight(
        PaidRunPreflight(
            run_kind="modal",
            artifact_dir=_Path(artifact_dir),
            cache_path=_Path(cache_path),
            split=split,
            seed=seed,
            model_id=MODEL_ID,
            backend=backend,
            estimated_dollar_cost=estimated_dollar_cost,
            cost_cap_dollars=cost_cap_dollars,
            user_authorized=user_authorized_paid_run,
        )
    )


def _ensure_model_cached(model_id: str) -> str:
    """Download once into the persistent HF cache and return the snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def _find_python_with_module(env: dict[str, str], module: str) -> str:
    import subprocess

    candidates = ["python3", "/usr/bin/python3", "python"]
    for candidate in candidates:
        probe = subprocess.run(
            [
                candidate,
                "-c",
                f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(f"No Python interpreter in the image can import {module}")


def _find_sglang_python(env: dict[str, str]) -> str:
    return _find_python_with_module(env, "sglang.launch_server")


def _run_sglang_python_json(script: str) -> dict:
    """Run a helper under the image Python that has SGLang/Torch installed."""
    return _run_module_python_json("sglang.launch_server", script)


def _run_module_python_json(module: str, script: str) -> dict:
    """Run a helper under the image Python that owns `module`."""
    import json
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/polaris/src"
    proc = subprocess.run(
        [_find_python_with_module(env, module), "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    marker = "POLARIS_JSON:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise RuntimeError(
        "SGLang helper failed or did not emit JSON\n"
        f"returncode={proc.returncode}\nstdout={proc.stdout[-4000:]}\nstderr={proc.stderr[-4000:]}"
    )


def _start_sglang_server(
    *,
    model_id: str,
    port: int = 30000,
    quantization: str | None = "fp8",
    kv_cache_dtype: str | None = "fp8_e5m2",
    max_running_requests: int = 64,
    max_total_tokens: int = 32768,
    mem_fraction_static: float = 0.90,
    attention_backend: str | None = None,
    sampling_backend: str | None = None,
    impl: str | None = None,
    disable_cuda_graph: bool = False,
) -> None:
    """Start local SGLang server for short R5 Modal smokes only."""
    import os
    import subprocess
    import time
    import urllib.request

    env = os.environ.copy()
    env["PYTHONPATH"] = "/polaris/src:" + env.get("PYTHONPATH", "")
    launch_python = _find_sglang_python(env)

    model_path = _ensure_model_cached(model_id)
    args = [
        launch_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-running-requests",
        str(max_running_requests),
        "--max-total-tokens",
        str(max_total_tokens),
        "--chunked-prefill-size",
        "8192",
        "--enable-custom-logit-processor",
        "--enable-metrics",
    ]
    if quantization is not None:
        args.extend(["--quantization", quantization])
    if kv_cache_dtype is not None:
        args.extend(["--kv-cache-dtype", kv_cache_dtype])
    if attention_backend is not None:
        args.extend(["--attention-backend", attention_backend])
    if sampling_backend is not None:
        args.extend(["--sampling-backend", sampling_backend])
    if impl is not None:
        args.extend(["--impl", impl])
    if disable_cuda_graph:
        args.append("--disable-cuda-graph")
    proc = subprocess.Popen(
        args,
        env=env,
        text=True,
    )
    for _ in range(450):
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise RuntimeError("SGLang failed to become healthy within startup timeout")


@app.function(
    gpu="A100-40GB",
    timeout=1200,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def smoke(
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """1 problem, greedy. Validates: image, GPU, model load, runner end-to-end."""
    _setup_paths()
    from pathlib import Path as _Path

    _require_modal_preflight(
        backend="hf",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.infra.serving.hf import RWSGenerator
    from polaris.io.manifest import compute_archive_hash
    from polaris.runners.math500 import run_condition

    sampler = RWSGenerator(seed=17)
    problems = load_math500_slice(0, 1)
    out = _Path("/polaris-runs/smoke-greedy-1p")
    metrics = run_condition(
        out_dir=out,
        condition="greedy",
        archive=MATH500_ARCHIVE_V1,
        cell_fitness={"direct_computation": 1.0},
        sampler=sampler,
        problems=problems,
        seed=17,
        archive_hash=compute_archive_hash(MATH500_ARCHIVE_V1),
        polaris_source_hash="modal-smoke",
        vendored_commits={"rws": "720a8e9d", "evalplus": "", "gepa": "", "dc": ""},
        preregistration_anchor="TODO.md#polaris-math500-v1",
    )
    runs_vol.commit()
    print(f"smoke metrics: {metrics}")
    return metrics


@app.function(
    image=sglang_image,
    gpu="H100",
    timeout=1800,
    volumes={
        "/cache/huggingface": hf_cache,
        "/root/.cache/sglang": sglang_cache,
        "/polaris-runs": runs_vol,
    },
)
def smoke_sglang_greedy(
    problem_idx: int = 0,
    max_new_tokens: int = 512,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """R5 cheap SGLang smoke: one greedy MATH500 problem, no long run."""
    _setup_paths()
    from polaris.config import MODEL_ID
    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.evals.verifiers.math import score_math
    from polaris.infra.serving.sglang import SGLangGenerator

    _require_modal_preflight(
        backend="sglang",
        split=(problem_idx, problem_idx + 1),
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    _start_sglang_server(model_id=MODEL_ID)
    problem = load_math500_slice(problem_idx, problem_idx + 1)[0]
    direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
    prompt_text = direct.compose(problem.prompt)
    sampler = SGLangGenerator(model_id=MODEL_ID)
    gen = sampler.generate_greedy(prompt_text, max_new_tokens=max_new_tokens)
    score = score_math(prompt_text + gen.generation, problem.answer)
    boxed_emitted = "\\boxed" in gen.generation
    summary = {
        "model_id": MODEL_ID,
        "problem_id": problem.problem_id,
        "passed": bool(score["passed"]),
        "boxed_emitted": boxed_emitted,
        "extracted": score["extracted"],
        "prompt_tokens": gen.prompt_token_count,
        "generation_tokens": gen.generation_token_count,
        "wall_clock_seconds": gen.wall_clock_seconds,
        "estimated_dollar_cost": gen.estimated_dollar_cost,
        "cost_target": "<0.10",
        "generation_preview": gen.generation[:500],
    }
    print(f"smoke_sglang_greedy: {summary}")
    return summary


@app.function(
    image=sglang_image,
    gpu="H100",
    timeout=2400,
    volumes={
        "/cache/huggingface": hf_cache,
        "/root/.cache/sglang": sglang_cache,
        "/polaris-runs": runs_vol,
    },
)
def smoke_sglang_parity(
    segment_len: int = 32,
    temperature: float = 0.25,
    impl: str = "sglang",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """R5 parity smoke: SGLang `score_segments` vs HF oracle on one tiny segment."""
    _setup_paths()
    from polaris.config import MODEL_ID

    _require_modal_preflight(
        backend="sglang",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    _start_sglang_server(
        model_id=MODEL_ID,
        quantization=None,
        kv_cache_dtype=None,
        max_running_requests=8,
        max_total_tokens=8192,
        mem_fraction_static=0.82,
        attention_backend="torch_native",
        sampling_backend="pytorch",
        impl=impl,
        disable_cuda_graph=True,
    )
    summary = _run_sglang_python_json(
        f"""
import json
import torch
import torch.nn.functional as F
import transformers
from transformers import LogitsProcessor

from polaris.config import MODEL_ID
from polaris.core.archive import MATH500_ARCHIVE_V1
from polaris.infra.serving.sglang import SGLangGenerator
from polaris.infra.serving.sglang_logits import (
    build_next_token_score_request,
    extract_output_token_id_logprob,
)

segment_len = {int(segment_len)}
temperature = {float(temperature)!r}
impl = {impl!r}
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=False, local_files_only=True
)
direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
problem_text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
answer = r"\\left( 3, \\frac{{\\pi}}{{2}} \\right)"
prompt_text = direct.compose(problem_text)
prefix_ids = tokenizer.encode(prompt_text)
target_text = " The final answer is \\\\boxed{{" + answer + "}}."
target_ids = tokenizer.encode(target_text, add_special_tokens=False)[:segment_len]

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
    local_files_only=True,
).to("cuda").eval()
full_ids = prefix_ids + target_ids
input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
rows = out.logits[0, len(prefix_ids) - 1 : len(full_ids) - 1, :].float()
targets = torch.tensor(target_ids, dtype=torch.long, device=rows.device)
hf_norm_tokens = (
    F.log_softmax(rows / temperature, dim=-1)
    .gather(-1, targets.unsqueeze(-1))
    .squeeze(-1)
)
hf_unnorm_tokens = (
    (1.0 / temperature)
    * F.log_softmax(rows, dim=-1)
    .gather(-1, targets.unsqueeze(-1))
    .squeeze(-1)
)
hf_lp_norm = float(hf_norm_tokens.sum().detach().cpu().item())
hf_lp_unnorm = float(hf_unnorm_tokens.sum().detach().cpu().item())

class ForcedTokenRecorder(LogitsProcessor):
    def __init__(self, target_ids, temperature):
        self.target_ids = [int(x) for x in target_ids]
        self.temperature = float(temperature)
        self.lp_norm = []
        self.lp_unnorm = []

    def __call__(self, input_ids, scores):
        pos = len(self.lp_norm)
        if pos >= len(self.target_ids):
            return scores
        target = self.target_ids[pos]
        row = scores[0].float()
        self.lp_norm.append(
            float(F.log_softmax(row / self.temperature, dim=-1)[target].detach().cpu())
        )
        self.lp_unnorm.append(
            float(
                (
                    (1.0 / self.temperature)
                    * F.log_softmax(row, dim=-1)[target]
                ).detach().cpu()
            )
        )
        forced = torch.full_like(scores, -torch.inf)
        forced[:, target] = 0.0
        return forced

forced_proc = ForcedTokenRecorder(target_ids, temperature)
forced_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
forced_output = model.generate(
    input_ids=forced_input_ids,
    attention_mask=torch.ones_like(forced_input_ids),
    max_new_tokens=len(target_ids),
    min_new_tokens=len(target_ids),
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    output_logits=True,
    logits_processor=[forced_proc],
)
hf_gen_token_ids = [
    int(x) for x in forced_output.sequences[0][len(prefix_ids) :].detach().cpu().tolist()
]
hf_gen_lp_norm = float(sum(forced_proc.lp_norm))
hf_gen_lp_unnorm = float(sum(forced_proc.lp_unnorm))

sgl = SGLangGenerator(model_id=MODEL_ID)
debug_req = build_next_token_score_request(
    prefix_ids, target_ids[0], temperature=temperature
)
debug_resp = sgl._transport("/generate", debug_req.to_payload())
debug_first = extract_output_token_id_logprob(debug_resp, target_ids[0])
sgl_score = sgl.score_segments([prefix_ids], [target_ids], temperature=temperature)
forward_vs_generate_diff = max(
    abs(hf_lp_norm - hf_gen_lp_norm),
    abs(hf_lp_unnorm - hf_gen_lp_unnorm),
)
sglang_vs_generate_diff = max(
    abs(hf_gen_lp_norm - sgl_score.lp_norm[0]),
    abs(hf_gen_lp_unnorm - sgl_score.lp_unnorm[0]),
)
sglang_vs_forward_diff = max(
    abs(hf_lp_norm - sgl_score.lp_norm[0]),
    abs(hf_lp_unnorm - sgl_score.lp_unnorm[0]),
)
summary = {{
    "model_id": MODEL_ID,
    "sglang_impl": impl,
    "segment_len": len(target_ids),
    "temperature": temperature,
    "hf_forward_lp_norm": hf_lp_norm,
    "hf_generate_lp_norm": hf_gen_lp_norm,
    "sglang_lp_norm": sgl_score.lp_norm[0],
    "hf_forward_lp_unnorm": hf_lp_unnorm,
    "hf_generate_lp_unnorm": hf_gen_lp_unnorm,
    "sglang_lp_unnorm": sgl_score.lp_unnorm[0],
    "hf_forward_vs_generate_max_abs_diff": forward_vs_generate_diff,
    "sglang_vs_hf_generate_max_abs_diff": sglang_vs_generate_diff,
    "sglang_vs_hf_forward_max_abs_diff": sglang_vs_forward_diff,
    "passed": forward_vs_generate_diff < 1e-3 and sglang_vs_generate_diff < 1e-3,
    "acceptance_threshold": 1e-3,
    "cost_target": "<0.50",
    "cuda_available": torch.cuda.is_available(),
    "target_ids_head": target_ids[:5],
    "hf_generate_token_ids_head": hf_gen_token_ids[:5],
    "hf_norm_head": [float(x) for x in hf_norm_tokens[:5].detach().cpu().tolist()],
    "hf_generate_norm_head": forced_proc.lp_norm[:5],
    "sglang_norm_head": sgl_score.lp_norm_tokens[0][:5],
    "hf_unnorm_head": [float(x) for x in hf_unnorm_tokens[:5].detach().cpu().tolist()],
    "hf_generate_unnorm_head": forced_proc.lp_unnorm[:5],
    "sglang_unnorm_head": sgl_score.lp_unnorm_tokens[0][:5],
    "debug_first_norm": debug_first,
    "debug_first_response": str(debug_resp)[:1200],
}}
print("POLARIS_JSON:" + json.dumps(summary, sort_keys=True))
"""
    )
    print(f"smoke_sglang_parity: {summary}")
    return summary


@app.function(
    image=image,
    timeout=600,
    volumes={"/polaris-runs": runs_vol},
)
def smoke_cache_replay() -> dict:
    """CPU cache replay smoke. No GPU; verifies second pass skips sampling."""
    _setup_paths()
    from dataclasses import dataclass
    from pathlib import Path as _Path

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.core.inference import polaris_inference
    from polaris.core.mixed_alpha import FIXED_ALPHA_4
    from polaris.io.trajectory_cache import TrajectoryCache

    @dataclass
    class _Gen:
        generation: str
        response_contains_prompt: bool = False
        prompt_token_count: int = 1
        generation_token_count: int = 1
        wall_clock_seconds: float = 1.0
        estimated_dollar_cost: float = 0.0
        acceptance_ratio: float | None = None

    class _Sampler:
        def __init__(self) -> None:
            self.calls = 0

        def generate_power(
            self,
            prompt_text,
            *,
            temperature,
            max_new_tokens,
            mcmc_steps=None,
            block_num=None,
        ):
            self.calls += 1
            return _Gen("\\boxed{1}")

        def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
            self.calls += 1
            return _Gen("\\boxed{1}")

    def scorer(full_response, reference):
        return {"score": 1.0, "passed": True}

    cache_path = _Path("/polaris-runs/r5-smokes/cache-replay.sqlite")
    if cache_path.exists():
        cache_path.unlink()
    cache = TrajectoryCache(cache_path)
    sampler = _Sampler()
    kwargs = dict(
        question="1+0",
        reference="1",
        archive=MATH500_ARCHIVE_V1,
        sampler=sampler,
        alpha_schedule=FIXED_ALPHA_4,
        total_samples=1,
        max_new_tokens=8,
        scorer=scorer,
        trajectory_cache=cache,
        cache_model_id="dummy",
        cache_track="math500",
        cache_problem_id="smoke",
        cache_seed=17,
    )
    polaris_inference(**kwargs)
    cold_calls = sampler.calls
    polaris_inference(**kwargs)
    replay_calls = sampler.calls - cold_calls
    summary = {
        "cache_path": str(cache_path),
        "cold_generation_calls": cold_calls,
        "replay_generation_calls": replay_calls,
        "passed": cold_calls == 1 and replay_calls == 0,
    }
    runs_vol.commit()
    print(f"smoke_cache_replay: {summary}")
    return summary


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 2,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def build_archive(
    dev_start: int = 75,
    dev_end: int = 100,
    seed: int = 17,
    samples_per_eval: int = 1,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> str:
    """Run MAP-Elites with --iterations 0: evaluate seeds on dev split, emit archive.json."""
    _setup_paths()
    import json
    from pathlib import Path as _Path

    _require_modal_preflight(
        backend="hf",
        split=(dev_start, dev_end),
        seed=seed,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        artifact_dir="/polaris-runs/2026-05-12-polaris-math500-v1/archive.json",
    )

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.core.descriptor import classify_trace
    from polaris.core.mapelite import run_mapelite
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.evals.verifiers.math import score_math
    from polaris.infra.serving.hf import RWSGenerator

    sampler = RWSGenerator(seed=seed)
    dev = load_math500_slice(dev_start, dev_end)
    grid = run_mapelite(
        seeds=MATH500_ARCHIVE_V1.entries,
        dev_set=dev,
        sampler=sampler,
        scorer=score_math,
        descriptor_fn=classify_trace,
        n_iterations=0,
        samples_per_eval=samples_per_eval,
    )
    archive = grid.freeze()
    out = _Path("/polaris-runs/2026-05-12-polaris-math500-v1/archive.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": archive.to_jsonable(),
        "cell_fitness": grid.cell_fitness(),
        "dev_slice": [dev_start, dev_end],
        "seed": seed,
        "samples_per_eval": samples_per_eval,
        "iterations": 0,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    runs_vol.commit()
    print(f"wrote {out}; cell_fitness={grid.cell_fitness()}")
    return str(out)


@app.function(
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def batched_mcmc_smoke(
    k_chains: int = 2,
    block_num: int = 4,
    mcmc_steps: int = 2,
    max_new_tokens: int = 512,
    problem_idx: int = 0,
    seed: int = 17,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Tiny batched-MCMC smoke. Defaults give ~2 min wall, ~$0.10.

    Validates `batched_mcmc_power_samp` on real GPU before scaling.
    """
    _setup_paths()
    import torch
    import transformers

    _require_modal_preflight(
        backend="hf",
        split=(problem_idx, problem_idx + 1),
        seed=seed,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.evals.verifiers.math import score_math
    from polaris.infra.mcmc import batched_mcmc_power_samp

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-7B", trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # FA2 needs install; sdpa is built-in fallback
    )

    problem = load_math500_slice(problem_idx, problem_idx + 1)[0]
    direct = next(e for e in MATH500_ARCHIVE_V1.entries if e.id == "direct")
    prompt_text = direct.compose(problem.prompt)
    prefix_ids = tokenizer.encode(prompt_text)

    print(
        f"prompt tokens: {len(prefix_ids)}, problem: {problem.problem_id}, answer: {problem.answer}"
    )
    result = batched_mcmc_power_samp(
        model,
        tokenizer,
        prefix_ids_batch=[list(prefix_ids) for _ in range(k_chains)],
        temp=0.25,
        mcmc_steps=mcmc_steps,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
        seed=seed,
    )

    summary = {
        "wall_seconds": result.wall_clock_seconds,
        "accept_ratios": result.acceptance_ratios,
        "passed": [],
    }
    for k, ids in enumerate(result.output_token_ids):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        score = score_math(prompt_text + text, problem.answer)
        summary["passed"].append(bool(score["passed"]))
        print(
            f"chain {k}: passed={score['passed']} gen_tokens={len(ids)} tail={text[-150:]!r}"
        )
    print(f"summary: {summary}")
    return summary


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 12,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def run_signal_pair(
    test_start: int = 0,
    test_end: int = 15,
    seed: int = 17,
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Matched-budget signal test: bon_temp1 (single prompt) vs bon_temp1_archive (k=4).

    No MCMC — pure low-temp T=1 BoN selection. Tests whether prompt-archive spread
    beats single prompt under identical sample budget. Cheap (~$15 total).
    """
    _setup_paths()
    from pathlib import Path as _Path

    _require_modal_preflight(
        backend="hf",
        split=(test_start, test_end),
        seed=seed,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    from polaris.core.archive import MATH500_ARCHIVE_V1
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.infra.serving.hf import RWSGenerator
    from polaris.io.manifest import compute_archive_hash
    from polaris.runners.math500 import run_condition

    sampler = RWSGenerator(seed=seed)
    problems = load_math500_slice(test_start, test_end)
    archive = MATH500_ARCHIVE_V1
    archive_hash = compute_archive_hash(archive)

    base_out = _Path(
        f"/polaris-runs/2026-05-12-signal/test-{test_start}-{test_end}-seed{seed}"
    )
    results = {}
    for cond in ("bon_temp1", "bon_temp1_archive"):
        metrics = run_condition(
            out_dir=base_out / cond,
            condition=cond,
            archive=archive,
            cell_fitness={},
            sampler=sampler,
            problems=problems,
            seed=seed,
            archive_hash=archive_hash,
            polaris_source_hash="modal-signal",
            vendored_commits={"rws": "720a8e9d", "evalplus": "", "gepa": "", "dc": ""},
            preregistration_anchor="TODO.md#polaris-math500-v1",
            split=(test_start, test_end),
        )
        results[cond] = metrics
        print(f"{cond}: {metrics}")
    runs_vol.commit()
    return results


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 12,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def run_one(
    condition: str,
    seed: int = 17,
    test_start: int = 0,
    test_end: int = 75,
    archive_path: str = "/polaris-runs/2026-05-12-polaris-math500-v1/archive.json",
    estimated_dollar_cost: float | None = None,
    cost_cap_dollars: float | None = None,
    user_authorized_paid_run: bool = False,
) -> dict:
    """Run one condition × one seed against the full test slice."""
    _setup_paths()
    import json
    from pathlib import Path as _Path

    _require_modal_preflight(
        backend="hf",
        split=(test_start, test_end),
        seed=seed,
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
    )

    from polaris.core.archive import FrozenArchive
    from polaris.evals.datasets.math500 import load_math500_slice
    from polaris.infra.serving.hf import RWSGenerator
    from polaris.io.manifest import compute_archive_hash
    from polaris.runners.math500 import run_condition

    archive_payload = json.loads(_Path(archive_path).read_text())
    archive = FrozenArchive.from_entries(archive_payload["entries"])
    cell_fitness = archive_payload.get("cell_fitness", {})

    sampler = RWSGenerator(seed=seed)
    problems = load_math500_slice(test_start, test_end)
    out = _Path(f"/polaris-runs/2026-05-12-polaris-math500-v1/{condition}/seed-{seed}/")
    metrics = run_condition(
        out_dir=out,
        condition=condition,
        archive=archive,
        cell_fitness=cell_fitness,
        sampler=sampler,
        problems=problems,
        seed=seed,
        archive_hash=compute_archive_hash(archive),
        polaris_source_hash="modal-v1",
        vendored_commits={"rws": "720a8e9d", "evalplus": "", "gepa": "", "dc": ""},
        preregistration_anchor="TODO.md#polaris-math500-v1",
        split=(test_start, test_end),
    )
    runs_vol.commit()
    print(f"{condition} seed={seed} metrics: {metrics}")
    return metrics


@app.function(
    gpu="A100-40GB",
    timeout=3600 * 6,
    volumes={"/cache/huggingface": hf_cache, "/polaris-runs": runs_vol},
)
def run_proicl_xfamily(
    user_authorized_paid_run: bool = False,
    estimated_dollar_cost: float | None = 5.0,
    cost_cap_dollars: float | None = 20.0,
    run_tag: str = "xfamily-v1",
    n_eval: int = 5,
    n_gepa_dev: int = 3,
    rollout_budget: int = 4,
    max_metric_calls: int = 16,
    archive_size: int = 2,
) -> dict:
    """Cross-family ProICL: GEPA trained on family_relationships, evaluated on graph_color.

    This is the held-out generalization test — GEPA never sees graph_color during
    prompt optimization. Run with:
        modal run scripts/modal_app.py::run_proicl_xfamily --user-authorized-paid-run true
    Pull results:
        modal volume get polaris-runs /proicl-xfamily runs/proicl_xfamily_modal/
    """
    import os
    import subprocess
    import sys
    from pathlib import Path as _Path

    _setup_paths()

    split = (20, 20 + n_eval)
    _require_modal_preflight(
        backend="hf",
        estimated_dollar_cost=estimated_dollar_cost,
        cost_cap_dollars=cost_cap_dollars,
        user_authorized_paid_run=user_authorized_paid_run,
        split=split,
        seed=17,
        artifact_dir=f"/polaris-runs/proicl-xfamily/{run_tag}",
        cache_path=f"/polaris-runs/proicl-xfamily/{run_tag}/trajectories.sqlite",
    )

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    # hf_transfer not installed in debian_slim image — disable to avoid crash in subprocesses
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    # No git binary in debian_slim — bypass vendored_commit() subprocess calls
    os.environ.setdefault("POLARIS_RWS_COMMIT", "modal-run")
    os.environ.setdefault("POLARIS_GEPA_COMMIT", "modal-run")
    os.environ.setdefault("POLARIS_EVALPLUS_COMMIT", "modal-run")
    os.environ.setdefault("POLARIS_DC_COMMIT", "modal-run")

    out_dir = _Path(f"/polaris-runs/proicl-xfamily/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "/polaris/scripts/run_proicl_signal.py",
        "--root", str(out_dir),
        "--tracks",
            "reasoning_gym_family_relationships",
            "reasoning_gym_graph_color_n10",
            "reasoning_gym_graph_color_n12",
        "--conditions",
            "base_greedy",
            "mcmc_only",
            "gepa_mcmc",
            "prorl_v2_greedy",
        "--archive-scope", "cross_family_curriculum",
        "--archive-train-tracks", "reasoning_gym_family_relationships",
        "--archive-heldout-tracks", "reasoning_gym_graph_color_n10", "reasoning_gym_graph_color_n12",
        "--eval-split", str(20), str(20 + n_eval),
        "--gepa-dev-split", str(0), str(n_gepa_dev),
        "--rollout-budget", str(rollout_budget),
        "--archive-size", str(archive_size),
        "--max-metric-calls", str(max_metric_calls),
        "--num-shards", "1",
        "--backend", "hf",
        "--run-kind", "modal",
        "--run-stage", "small_real_slice",
        "--cost-cap-dollars", str(cost_cap_dollars),
        "--reflection-provider", "local-hf",
        "--reflection-model-id", "Qwen/Qwen2.5-7B-Instruct",
        "--skip-prefetch",
        "--skip-smoke",
        "--gpus", "0",
    ]

    print("Launching:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd="/polaris", text=True, check=False)

    runs_vol.commit()

    decomp_path = out_dir / "analysis" / "proicl_decomposition.json"
    if decomp_path.exists():
        import json
        decomp = json.loads(decomp_path.read_text())
        print("=== DECOMPOSITION ===")
        print(json.dumps(decomp, indent=2))
        return decomp
    return {"returncode": result.returncode, "error": "decomposition not written"}
