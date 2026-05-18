from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.core.archive import FrozenArchive, PromptEntry
from polaris.evals.datasets.math500 import Problem
from polaris.proicl.analysis import compute_proicl_decomposition
from polaris.proicl.analysis import write_proicl_decomposition_by_track
from polaris.proicl.archive import build_cross_task_curriculum_archive
from polaris.proicl.launcher import (
    build_signal_cells,
    cell_complete,
    required_artifacts,
    run_cells,
    run_condition_command,
)
from polaris.proicl import launcher as launcher_mod
from polaris.proicl.run_graph import (
    PROICL_PRIMARY_CONDITIONS,
    build_proicl_run_graph,
)
from polaris.runners.condition_runner import run_condition
from scripts.run_proicl_signal import (
    _gepa_archive_command,
    _partition_cells,
    configure_flow_cache_env,
)


def test_proicl_run_graph_is_the_clean_four_cell_factorial_plus_prorl_reference():
    plan = build_proicl_run_graph(
        root="/runs/proicl",
        tracks=("reasoning_gym_boxnet",),
        problem_count=100,
        rollout_budget=64,
        num_shards=2,
    )

    assert PROICL_PRIMARY_CONDITIONS == (
        "base_greedy",
        "mcmc_only",
        "gepa_only",
        "gepa_mcmc",
        "gepa_mcmc_memory",
        "prorl_v2_greedy",
    )
    assert {cell["proicl_condition"] for cell in plan["cells"]} == set(
        PROICL_PRIMARY_CONDITIONS
    )

    base = next(c for c in plan["cells"] if c["proicl_condition"] == "base_greedy")
    mcmc = next(c for c in plan["cells"] if c["proicl_condition"] == "mcmc_only")
    gepa = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_only")
    composed = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc")
    memory = next(c for c in plan["cells"] if c["proicl_condition"] == "gepa_mcmc_memory")
    prorl = next(c for c in plan["cells"] if c["proicl_condition"] == "prorl_v2_greedy")

    assert base["model_key"] == "deepseek-r1-distill-qwen-1.5b"
    assert base["runtime_condition"] == "greedy"
    assert base["condition"] == "greedy"
    assert base["archive_kind"] == "direct"
    assert base["rollout_budget"] == 1

    assert mcmc["model_key"] == "deepseek-r1-distill-qwen-1.5b"
    assert mcmc["runtime_condition"] == "single_prompt_power"
    assert mcmc["condition"] == "single_prompt_power"
    assert mcmc["archive_kind"] == "direct"
    assert mcmc["uses_power_sampling"] is True
    assert mcmc["uses_gepa_archive"] is False

    assert gepa["runtime_condition"] == "gepa_only"
    assert gepa["archive_kind"] == "proicl_cross_task_gepa"
    assert gepa["uses_power_sampling"] is False
    assert gepa["uses_gepa_archive"] is True

    assert composed["runtime_condition"] == "proicl_gepa_mcmc"
    assert composed["archive_kind"] == "proicl_cross_task_gepa"
    assert composed["rollout_budget"] == 64
    assert composed["uses_memory"] is False

    assert memory["runtime_condition"] == "proicl_gepa_mcmc_memory"
    assert memory["condition"] == "proicl_gepa_mcmc_memory"
    assert memory["uses_memory"] is True
    assert memory["memory_mode"] == "distilled_strategies"
    assert memory["memory_store_path"].endswith(
        "reasoning_gym_boxnet-gepa_mcmc_memory-shard-0.sqlite"
    )

    assert prorl["model_key"] == "nemotron-prorl-v2"
    assert prorl["runtime_condition"] == "greedy"
    assert prorl["condition"] == "greedy"
    assert prorl["slow_weight_reference"] is True


def test_proicl_signal_launcher_materializes_executable_cells(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 40),
        rollout_budget=8,
        num_shards=2,
        memory_num_shards=1,
    )

    by_condition = {}
    for cell in cells:
        by_condition.setdefault(cell.proicl_condition, []).append(cell)

    assert set(by_condition) == {
        "base_greedy",
        "mcmc_only",
        "gepa_only",
        "gepa_mcmc",
        "gepa_mcmc_memory",
        "prorl_v2_greedy",
    }
    assert len(by_condition["mcmc_only"]) == 2
    assert len(by_condition["gepa_mcmc_memory"]) == 1

    memory = by_condition["gepa_mcmc_memory"][0]
    assert memory.condition == "proicl_gepa_mcmc_memory"
    assert memory.memory_store_path is not None
    assert "--memory-mode" in memory.extra_args
    assert "memory.sqlite" in required_artifacts(memory)
    assert "archive_build_manifest.json" in required_artifacts(memory)
    assert "archive.json" in required_artifacts(memory)
    assert "rollouts.json" in required_artifacts(memory)

    assert cell_complete(memory) is False


def test_proicl_signal_cell_partitioning_is_disjoint_and_complete(tmp_path):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet", "reasoning_gym_graph_color"),
        split=(20, 40),
        rollout_budget=8,
        num_shards=2,
        memory_num_shards=1,
    )

    partitions = [
        _partition_cells(cells, stride=4, offset=offset)
        for offset in range(4)
    ]
    flattened = [cell for partition in partitions for cell in partition]

    assert len(flattened) == len(cells)
    assert {id(cell) for cell in flattened} == {id(cell) for cell in cells}
    assert all(partitions)


def test_proicl_signal_launcher_rejects_parallel_memory_shards(tmp_path):
    with pytest.raises(ValueError, match="memory_num_shards=1"):
        build_signal_cells(
            root=tmp_path / "proicl",
            tracks=("reasoning_gym_boxnet",),
            split=(20, 40),
            rollout_budget=8,
            num_shards=2,
            memory_num_shards=2,
        )


def test_proicl_launcher_reuses_the_gpu_that_finished(tmp_path, monkeypatch):
    cells = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 24),
        rollout_budget=2,
        num_shards=3,
        memory_num_shards=1,
    )[:3]
    launches: list[str] = []
    durations = [3, 1, 1]
    completed: set[str] = set()
    stale = Path(cells[0].artifact_dir) / "candidates.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale partial row\n", encoding="utf-8")

    class FakeProc:
        def __init__(self, ticks: int, artifact_dir: str):
            self.ticks = ticks
            self.artifact_dir = artifact_dir

        def poll(self):
            self.ticks -= 1
            if self.ticks > 0:
                return None
            completed.add(self.artifact_dir)
            return 0

    def fake_popen(cmd, *, cwd, env, stdout, stderr):
        launches.append(env["CUDA_VISIBLE_DEVICES"])
        return FakeProc(durations.pop(0), cmd[1])

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        launcher_mod,
        "run_condition_command",
        lambda **kwargs: ["fake", kwargs["cell"].artifact_dir],
    )
    monkeypatch.setattr(
        launcher_mod,
        "cell_complete",
        lambda cell: cell.artifact_dir in completed,
    )

    run_cells(
        repo_root=tmp_path,
        cells=cells,
        gpus=["0", "1"],
        events_path=tmp_path / "events.jsonl",
        backend="hf",
        local_files_only=False,
        cost_cap_dollars=1.0,
        estimated_dollar_cost=0.1,
        estimated_wall_clock_seconds=10,
        run_kind="local",
        run_stage="smoke",
        max_new_tokens=8,
    )

    assert launches == ["0", "1", "1"]
    assert not stale.exists()


def test_proicl_condition_command_propagates_vllm_scoring_mode(tmp_path):
    cell = build_signal_cells(
        root=tmp_path / "proicl",
        tracks=("reasoning_gym_boxnet",),
        split=(20, 21),
        rollout_budget=2,
        num_shards=1,
        memory_num_shards=1,
    )[1]

    cmd = run_condition_command(
        repo_root=Path("/repo"),
        cell=cell,
        backend="vllm",
        local_files_only=True,
        polaris_source_hash="hash",
        cost_cap_dollars=1.0,
        estimated_dollar_cost=0.1,
        estimated_wall_clock_seconds=10.0,
        run_kind="local",
        run_stage="smoke",
        max_new_tokens=64,
        mcmc_steps=2,
        mcmc_block_num=4,
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.7,
        vllm_max_model_len=2048,
        vllm_scoring_mode="native_segment",
        vllm_parity_artifact="runs/calibration/calibration_summary.json",
    )

    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "vllm"
    assert cmd[cmd.index("--vllm-scoring-mode") + 1] == "native_segment"
    assert cmd[cmd.index("--vllm-dtype") + 1] == "float32"
    assert cmd[cmd.index("--vllm-model-impl") + 1] == "transformers"
    assert cmd[cmd.index("--vllm-gpu-memory-utilization") + 1] == "0.7"
    assert cmd[cmd.index("--vllm-max-model-len") + 1] == "2048"
    assert cmd[cmd.index("--mcmc-steps") + 1] == "2"
    assert cmd[cmd.index("--mcmc-block-num") + 1] == "4"
    assert (
        cmd[cmd.index("--vllm-parity-artifact") + 1]
        == "runs/calibration/calibration_summary.json"
    )
    assert "--local-files-only" in cmd


def test_proicl_signal_flow_cache_overrides_workspace_defaults(tmp_path):
    env = {
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    }

    configure_flow_cache_env(
        env,
        run_kind="flow",
        mnt_local=tmp_path,
        cache_root=tmp_path / "proicl-cache",
    )

    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["HF_HUB_CACHE"].startswith(str(tmp_path))
    assert env["HUGGINGFACE_HUB_CACHE"] == env["HF_HUB_CACHE"]
    assert env["PIP_CACHE_DIR"].startswith(str(tmp_path))


def test_proicl_gepa_archive_command_can_pin_single_gpu(tmp_path):
    args = SimpleNamespace(
        env_file=tmp_path / ".env",
        local_files_only=False,
        backend="hf",
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.85,
        vllm_max_model_len=None,
        vllm_scoring_mode="forced_decode_v0",
        vllm_parity_artifact=None,
    )

    cmd = _gepa_archive_command(
        args=args,
        out=tmp_path / "archive",
        tracks=["reasoning_gym_boxnet", "reasoning_gym_graph_color"],
        dev_split=(0, 6),
        archive_size=8,
        max_metric_calls=64,
        sampler_max_new_tokens=512,
        reflection_max_new_tokens=512,
        cuda_visible_devices="0",
    )

    assert "--cuda-visible-devices" in cmd
    assert cmd[cmd.index("--cuda-visible-devices") + 1] == "0"
    assert "--live-gepa" in cmd
    assert "--reflection-provider" in cmd
    assert cmd[cmd.index("--reflection-provider") + 1] == "xai"


def test_proicl_gepa_archive_command_propagates_vllm_sampler_backend(tmp_path):
    args = SimpleNamespace(
        env_file=tmp_path / ".env",
        local_files_only=False,
        backend="vllm",
        vllm_dtype="float32",
        vllm_model_impl="transformers",
        vllm_gpu_memory_utilization=0.65,
        vllm_max_model_len=4096,
        vllm_scoring_mode="forced_decode_v0",
        vllm_parity_artifact=tmp_path / "calibration_summary.json",
    )

    cmd = _gepa_archive_command(
        args=args,
        out=tmp_path / "archive",
        tracks=["reasoning_gym_boxnet"],
        dev_split=(0, 6),
        archive_size=8,
        max_metric_calls=64,
        sampler_max_new_tokens=512,
        reflection_max_new_tokens=512,
        cuda_visible_devices="0",
    )

    assert cmd[cmd.index("--sampler-backend") + 1] == "vllm"
    assert cmd[cmd.index("--vllm-dtype") + 1] == "float32"
    assert cmd[cmd.index("--vllm-gpu-memory-utilization") + 1] == "0.65"
    assert cmd[cmd.index("--vllm-max-model-len") + 1] == "4096"
    assert cmd[cmd.index("--vllm-parity-artifact") + 1] == str(
        tmp_path / "calibration_summary.json"
    )


def test_proicl_decomposition_reports_rf_and_slow_residual():
    report = compute_proicl_decomposition(
        {
            "base": 0.10,
            "mcmc_only": 0.25,
            "gepa_only": 0.34,
            "gepa_mcmc": 0.42,
            "gepa_mcmc_memory": 0.50,
            "prorl_v2_greedy": 0.70,
        }
    )

    assert report["A_base"] == 0.10
    assert report["A_mcmc"] == 0.25
    assert report["A_gepa"] == 0.34
    assert report["A_gepa_mcmc"] == 0.42
    assert report["A_memory"] == 0.50
    assert report["A_prorl_v2"] == 0.70
    assert report["rf_valid"] is True
    assert report["RF_mcmc"] == pytest.approx(0.25)
    assert report["RF_gepa"] == pytest.approx(0.4)
    assert report["RF_gepa_mcmc"] == pytest.approx(0.5333333333)
    assert report["RF_memory"] == pytest.approx(0.6666666667)
    assert report["slow_weight_residual"] == pytest.approx(0.28)
    assert report["memory_gain"] == pytest.approx(0.08)


def test_proicl_decomposition_tolerates_nonpositive_rf_denominator():
    report = compute_proicl_decomposition(
        {
            "base": 0.50,
            "mcmc_only": 0.55,
            "gepa_only": 0.55,
            "gepa_mcmc": 0.60,
            "gepa_mcmc_memory": 0.60,
            "prorl_v2_greedy": 0.50,
        }
    )

    assert report["rf_valid"] is False
    assert report["rf_denominator"] == pytest.approx(0.0)
    assert report["RF_mcmc"] is None
    assert "analysis_warning" in report


def test_proicl_aggregate_reads_shard_metrics(tmp_path):
    values = {
        "base_greedy": [0.10, 0.20],
        "mcmc_only": [0.25, 0.35],
        "gepa_only": [0.34, 0.44],
        "gepa_mcmc": [0.42, 0.52],
        "gepa_mcmc_memory": [0.50, 0.60],
        "prorl_v2_greedy": [0.70, 0.80],
    }
    root = tmp_path / "proicl"
    for condition, accuracies in values.items():
        for shard_id, accuracy in enumerate(accuracies):
            path = root / "runs" / "reasoning_gym_boxnet" / condition / f"shard-{shard_id}"
            path.mkdir(parents=True)
            (path / "metrics.json").write_text(
                json.dumps({"accuracy": accuracy, "n_problems": 5}),
                encoding="utf-8",
            )

    report = write_proicl_decomposition_by_track(
        root=root,
        tracks=("reasoning_gym_boxnet",),
        out_dir=tmp_path / "analysis",
    )

    decomp = report["reasoning_gym_boxnet"]["decomposition"]
    assert decomp["A_base"] == pytest.approx(0.15)
    assert decomp["A_prorl_v2"] == pytest.approx(0.75)
    assert (tmp_path / "analysis" / "proicl_decomposition.json").exists()
    md = (tmp_path / "analysis" / "proicl_decomposition.md")
    assert md.exists()
    text = md.read_text()
    assert "discovery_gain" in text
    assert "composition_gain" in text


def test_cross_task_curriculum_archive_writes_k16_gepa_artifacts(tmp_path):
    out = tmp_path / "archive"
    payload = build_cross_task_curriculum_archive(
        out_dir=out,
        tracks=("reasoning_gym_boxnet", "reasoning_gym_graph_color"),
        dev_split=(0, 2),
        archive_size=16,
        dry_run=True,
        max_metric_calls=32,
        reflection_provider="none",
    )

    archive = json.loads((out / "archive.json").read_text())
    manifest = json.loads((out / "archive_build_manifest.json").read_text())

    assert payload["archive_build_id"].startswith("proicl-cross-task-gepa")
    assert len(archive["entries"]) == 16
    assert manifest["archive_scope"] == "cross_task_reasoning_gym"
    assert manifest["gepa"]["dry_run"] is True
    assert manifest["gepa"]["max_metric_calls"] == 32
    assert manifest["leakage_policy"]["no_prorl_or_brorl_traces"] is True


def test_live_cross_task_gepa_uses_vendored_gepa_without_external_package(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "gepa", None)

    rows = [
        Problem(
            problem_id="rg-dev-0",
            prompt="Return the exact word pass.",
            answer="pass",
            source="unit",
        )
    ]
    monkeypatch.setattr(
        "polaris.proicl.archive._load_rows",
        lambda *, tracks, split, dry_run: (rows, "optimizer_dev"),
    )

    class Sampler:
        def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
            text = "pass" if "Use the exact requested word" in prompt_text else "fail"
            return SimpleNamespace(
                generation=text,
                response_contains_prompt=False,
                prompt_token_count=len(prompt_text.split()),
                generation_token_count=1,
                wall_clock_seconds=0.0,
                estimated_dollar_cost=0.0,
            )

    class ReflectionLM:
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        def __call__(self, prompt):
            self.total_tokens_in += 1
            self.total_tokens_out += 1
            return "```Use the exact requested word and place it inside <answer> tags.```"

    def scorer(response: str, reference: str) -> dict:
        passed = response.strip().endswith("pass")
        return {"score": 1.0 if passed else 0.0, "passed": passed}

    payload = build_cross_task_curriculum_archive(
        out_dir=tmp_path / "archive",
        tracks=("reasoning_gym_boxnet",),
        dev_split=(0, 1),
        archive_size=2,
        dry_run=False,
        max_metric_calls=4,
        reflection_provider="local_hf",
        sampler=Sampler(),
        scorer=scorer,
        reflection_lm=ReflectionLM(),
    )

    manifest = json.loads(
        (tmp_path / "archive" / "archive_build_manifest.json").read_text()
    )
    assert payload["gepa"]["dry_run"] is False
    assert manifest["reflection"]["provider"] == "local_hf"
    assert manifest["reflection"]["usage"]["total_tokens_out"] >= 1
    archive = json.loads((tmp_path / "archive" / "archive.json").read_text())
    assert len(archive["entries"]) == 2


class _FakeSampler:
    def generate_greedy(self, prompt_text, max_new_tokens):
        return self._gen(prompt_text)

    def generate_low_temp(self, prompt_text, *, temperature, max_new_tokens):
        return self._gen(prompt_text)

    def generate_power(
        self, prompt_text, *, temperature, max_new_tokens, mcmc_steps=None, block_num=None
    ):
        return self._gen(prompt_text, acceptance_ratio=1.0)

    def _gen(self, prompt_text, acceptance_ratio=None):
        return SimpleNamespace(
            generation=" solved",
            response_contains_prompt=False,
            prompt_token_count=len(prompt_text.split()),
            generation_token_count=1,
            wall_clock_seconds=0.0,
            estimated_dollar_cost=0.0,
            acceptance_ratio=acceptance_ratio,
            token_ids=[],
            logprobs_norm=[],
            logprobs_unnorm=[],
        )


def test_proicl_gepa_mcmc_memory_writes_memory_ledgers(tmp_path):
    archive = FrozenArchive(
        entries=(
            PromptEntry(
                id="direct",
                prefix="Q: ",
                suffix=" A:",
                descriptor_hint="direct",
            ),
        )
    )
    problem = Problem(
        problem_id="toy-0",
        prompt="2+2?",
        answer="4",
        source="unit",
    )

    metrics = run_condition(
        track="math500",
        split=(0, 1),
        model_key="deepseek-r1-distill-qwen-1.5b",
        out_dir=tmp_path / "run",
        condition="proicl_gepa_mcmc_memory",
        archive=archive,
        cell_fitness={"direct": 1.0},
        sampler=_FakeSampler(),
        seed=17,
        archive_hash="archive",
        polaris_source_hash="test",
        vendored_commits={},
        preregistration_anchor="TODO.PROICL.md",
        model_id="fake",
        model_revision="main",
        model_revision_commit="fake",
        problems=[problem],
        budget_override=1,
        max_new_tokens=8,
        archive_build_id="proicl-cross-task-gepa-test",
        memory_build_id="proicl-memory-test",
        serving_backend_metadata={
            "backend": "vllm",
            "vllm_scoring_mode": "native_segment",
            "vllm_version": "unit",
            "vllm_commit": "abc123",
            "dtype": "float32",
            "model_impl": "transformers",
            "tokenizer_revision": "main",
            "parity_artifact_path": "runs/calibration/calibration_summary.json",
        },
    )

    assert metrics["condition"] == "proicl_gepa_mcmc_memory"
    assert metrics["alpha_policy_id"] == "fixed_alpha_4"
    assert metrics["backend"] == "vllm"
    assert metrics["vllm_scoring_mode"] == "native_segment"
    assert (tmp_path / "run" / "memory.sqlite").exists()
    assert (tmp_path / "run" / "memory_events.jsonl").exists()
    assert (tmp_path / "run" / "archive_build_manifest.json").exists()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["config"]["serving_backend"]["vllm_version"] == "unit"
    assert (
        manifest["config"]["vllm_parity_artifact"]
        == "runs/calibration/calibration_summary.json"
    )
