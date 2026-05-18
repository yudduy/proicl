from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from polaris.infra.serving.vllm import (
    VLLMForcedTokenProcessor,
    VLLMGenerator,
    VLLMSamplingRecorderProcessor,
)


def test_vllm_forced_token_processor_records_and_forces_logits():
    proc = VLLMForcedTokenProcessor([2], temperature=0.5)
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])

    forced = proc([9], [], logits)

    assert int(torch.argmax(forced).item()) == 2
    assert torch.isneginf(forced[0])
    assert proc.lp_norm == pytest.approx(
        [float(F.log_softmax(logits.float() / 0.5, dim=-1)[2])]
    )
    assert proc.lp_unnorm == pytest.approx(
        [float(2.0 * F.log_softmax(logits.float(), dim=-1)[2])]
    )


def test_vllm_sampling_recorder_samples_records_and_forces_logits():
    proc = VLLMSamplingRecorderProcessor(temperature=0.5, seed=0)
    logits = torch.tensor([-100.0, -100.0, 100.0, -100.0])

    forced = proc([9], [], logits)

    assert int(torch.argmax(forced).item()) == 2
    assert proc.token_ids == [2]
    assert proc.lp_norm == pytest.approx(
        [float(F.log_softmax(logits.float() / 0.5, dim=-1)[2])]
    )
    assert proc.lp_unnorm == pytest.approx(
        [float(2.0 * F.log_softmax(logits.float(), dim=-1)[2])]
    )


def test_vllm_low_temp_batch_uses_one_generate_call_and_stable_offsets(monkeypatch):
    seen_batch_sizes = []
    seen_processor_seeds = []
    seen_temperatures = []
    seen_ignore_eos = []
    seen_min_tokens = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params, use_tqdm=False):
            seen_batch_sizes.append(len(prompts))
            outputs = []
            for row_idx, (prompt, params) in enumerate(zip(prompts, sampling_params)):
                seen_temperatures.append(params.temperature)
                seen_ignore_eos.append(params.ignore_eos)
                seen_min_tokens.append(getattr(params, "min_tokens", None))
                token_ids = []
                processors = getattr(params, "logits_processors", None) or []
                if processors:
                    processor = processors[0]
                    for _ in range(params.max_tokens):
                        logits = torch.full((16,), -100.0)
                        logits[row_idx + 1] = 100.0
                        logits = processor(prompt["prompt_token_ids"], token_ids, logits)
                        token_ids.append(int(torch.argmax(logits).item()))
                else:
                    seen_processor_seeds.append(params.seed)
                    token_ids = [row_idx + 1]
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]))
            return outputs

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 15

        def encode(self, text):
            return [1, len(text)]

        def decode(self, token_ids, skip_special_tokens=True):
            return ",".join(str(x) for x in token_ids)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    gen = VLLMGenerator(model_id="fake-model", seed=17)

    outs = gen.generate_low_temp_batch(
        ["a", "bb", "ccc"],
        temperature=0.5,
        max_new_tokens=2,
        seed_base=100,
        seed_offsets=[9, 2, 7],
    )

    assert seen_batch_sizes == [3]
    assert seen_processor_seeds == [109, 102, 107]
    assert seen_temperatures == [0.5, 0.5, 0.5]
    assert seen_ignore_eos == [False, False, False]
    assert seen_min_tokens == [None, None, None]
    assert [out.token_ids for out in outs] == [[1], [2], [3]]
    assert [out.generation for out in outs] == ["1", "2", "3"]
    assert all(out.logprobs_norm == [] for out in outs)
    assert all(out.logprobs_unnorm == [] for out in outs)


def test_vllm_generate_power_batch_batches_drafts_and_proposals(monkeypatch):
    seen_batch_sizes = []
    seen_temperatures = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params, use_tqdm=False):
            seen_batch_sizes.append(len(prompts))
            outputs = []
            for row_idx, (prompt, params) in enumerate(zip(prompts, sampling_params)):
                seen_temperatures.append(params.temperature)
                token_ids = []
                processor = params.logits_processors[0]
                for _ in range(params.max_tokens):
                    logits = torch.full((16,), -100.0)
                    logits[row_idx + 1] = 100.0
                    logits = processor(prompt["prompt_token_ids"], token_ids, logits)
                    token_ids.append(int(torch.argmax(logits).item()))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]))
            return list(reversed(outputs))

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 15

        def encode(self, text):
            return [1, len(text)]

        def decode(self, token_ids, skip_special_tokens=True):
            return ",".join(str(x) for x in token_ids)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    gen = VLLMGenerator(model_id="fake-model", seed=17, fused_sampling_recorder=True)

    outs = gen.generate_power_batch(
        ["a", "bb", "ccc"],
        temperature=0.5,
        max_new_tokens=4,
        block_num=2,
        mcmc_steps=1,
        seed_base=100,
        seed_offsets=[0, 10, 20],
    )

    assert seen_batch_sizes == [3, 3, 3, 3]
    assert seen_temperatures == [1.0] * 12
    assert [out.token_ids for out in outs] == [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    assert all(out.generation_token_count == 4 for out in outs)
    assert all(len(out.logprobs_norm or []) == 4 for out in outs)
    assert all(len(out.logprobs_unnorm or []) == 4 for out in outs)
    assert all(out.acceptance_ratio == pytest.approx(1.0) for out in outs)


def test_vllm_score_segments_uses_forced_decode(monkeypatch):
    fake_llm_holder = {}
    seen_temperatures = []
    seen_min_tokens = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.reset_count = 0
            self.llm_engine = SimpleNamespace(reset_prefix_cache=self._reset_prefix_cache)
            fake_llm_holder["llm"] = self

        def _reset_prefix_cache(self):
            self.reset_count += 1

        def generate(self, prompts, sampling_params, use_tqdm=False):
            outputs = []
            params_list = (
                sampling_params
                if isinstance(sampling_params, list)
                else [sampling_params for _ in prompts]
            )
            for prompt, params in zip(prompts, params_list):
                seen_temperatures.append(params.temperature)
                seen_min_tokens.append(params.min_tokens)
                prompt_ids = prompt["prompt_token_ids"]
                token_ids = []
                for _ in range(params.max_tokens):
                    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
                    for proc in getattr(params, "logits_processors", []) or []:
                        logits = proc(prompt_ids, token_ids, logits)
                    token_ids.append(int(torch.argmax(logits).item()))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]))
            return outputs

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 3

        def encode(self, text):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(x) for x in token_ids)

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeTokenizer()
        )
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    gen = VLLMGenerator(model_id="fake-model", dtype="float32", model_impl="transformers")
    scores = gen.score_segments([[10]], [[2, 1]], temperature=0.5)

    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected_norm = [
        float(F.log_softmax(logits / 0.5, dim=-1)[2]),
        float(F.log_softmax(logits / 0.5, dim=-1)[1]),
    ]
    expected_unnorm = [
        float(2.0 * F.log_softmax(logits, dim=-1)[2]),
        float(2.0 * F.log_softmax(logits, dim=-1)[1]),
    ]
    assert len(scores.lp_norm_tokens) == 1
    assert len(scores.lp_unnorm_tokens) == 1
    assert scores.lp_norm_tokens[0] == pytest.approx(expected_norm)
    assert scores.lp_unnorm_tokens[0] == pytest.approx(expected_unnorm)
    assert scores.lp_norm == pytest.approx([sum(expected_norm)])
    assert scores.lp_unnorm == pytest.approx([sum(expected_unnorm)])
    assert seen_temperatures == [1.0]
    assert seen_min_tokens == [0]
    assert fake_llm_holder["llm"].kwargs["enable_prefix_caching"] is True
    assert fake_llm_holder["llm"].reset_count == 2
    assert fake_llm_holder["llm"].kwargs["dtype"] == "float32"
    assert fake_llm_holder["llm"].kwargs["model_impl"] == "transformers"


def test_vllm_score_segments_can_skip_prefix_cache_reset(monkeypatch):
    fake_llm_holder = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.reset_count = 0
            self.llm_engine = SimpleNamespace(reset_prefix_cache=self._reset_prefix_cache)
            fake_llm_holder["llm"] = self

        def _reset_prefix_cache(self):
            self.reset_count += 1

        def generate(self, prompts, sampling_params, use_tqdm=False):
            outputs = []
            params_list = (
                sampling_params
                if isinstance(sampling_params, list)
                else [sampling_params for _ in prompts]
            )
            for prompt, params in zip(prompts, params_list):
                token_ids = []
                for _ in range(params.max_tokens):
                    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
                    for proc in getattr(params, "logits_processors", []) or []:
                        logits = proc(prompt["prompt_token_ids"], token_ids, logits)
                    token_ids.append(int(torch.argmax(logits).item()))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]))
            return outputs

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 3

        def encode(self, text):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(x) for x in token_ids)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    gen = VLLMGenerator(
        model_id="fake-model",
        enable_prefix_caching=False,
        reset_prefix_cache_for_scoring=True,
    )
    gen.score_segments([[10]], [[2]], temperature=0.5)

    assert fake_llm_holder["llm"].kwargs["enable_prefix_caching"] is False
    assert fake_llm_holder["llm"].reset_count == 0


def test_vllm_score_segments_batches_non_empty_requests(monkeypatch):
    seen_batch_sizes = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params, use_tqdm=False):
            seen_batch_sizes.append(len(prompts))
            outputs = []
            for prompt, params in zip(prompts, sampling_params):
                prompt_ids = prompt["prompt_token_ids"]
                token_ids = []
                for _ in range(params.max_tokens):
                    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
                    for proc in params.logits_processors:
                        logits = proc(prompt_ids, token_ids, logits)
                    token_ids.append(int(torch.argmax(logits).item()))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]))
            return outputs

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text):
            return []

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    gen = VLLMGenerator(model_id="fake-model")

    scores = gen.score_segments([[1], [2], [3]], [[2], [], [1, 2]], temperature=0.5)

    assert seen_batch_sizes == [2]
    assert scores.lp_norm_tokens[1] == []
    assert scores.lp_unnorm_tokens[1] == []
    assert scores.lp_norm[1] == 0.0
    assert len(scores.lp_norm_tokens[0]) == 1
    assert len(scores.lp_norm_tokens[2]) == 2


def test_vllm_score_segments_rejects_length_mismatch(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text):
            return []

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=lambda **kwargs: SimpleNamespace(generate=lambda **kwargs: []),
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    gen = VLLMGenerator(model_id="fake-model")

    with pytest.raises(ValueError, match="length mismatch"):
        gen.score_segments([[1]], [[2], [3]], temperature=0.5)


def test_vllm_native_segment_score_sequences_adapter(monkeypatch):
    seen = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def score_sequences(self, *, prompt_token_ids, target_token_ids, temperatures):
            seen["prompt_token_ids"] = prompt_token_ids
            seen["target_token_ids"] = target_token_ids
            seen["temperatures"] = temperatures
            return [
                {
                    "target_token_ids": target_token_ids[0],
                    "lp_base": [-2.0, -4.0],
                    "lp_temp": [-1.25, -3.25],
                }
            ]

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 99

        def encode(self, text):
            return [1]

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )

    gen = VLLMGenerator(
        model_id="fake-model",
        dtype="float32",
        model_impl="transformers",
        scoring_mode="native_segment",
        parity_artifact_path="runs/calibration/calibration_summary.json",
    )
    scores = gen.score_segments([[10], [11]], [[2, 1], []], temperature=0.5)

    assert seen == {
        "prompt_token_ids": [[10]],
        "target_token_ids": [[2, 1]],
        "temperatures": [0.5],
    }
    assert scores.lp_norm_tokens == [[-1.25, -3.25], []]
    assert scores.lp_unnorm_tokens == [[-4.0, -8.0], []]
    assert scores.lp_norm == pytest.approx([-4.5, 0.0])
    assert scores.lp_unnorm == pytest.approx([-12.0, 0.0])
    metadata = gen.runtime_metadata()
    assert metadata["backend"] == "vllm"
    assert metadata["vllm_scoring_mode"] == "native_segment"
    assert metadata["dtype"] == "float32"
    assert metadata["model_impl"] == "transformers"
    assert metadata["native_segment_available"] is True
    assert metadata["parity_artifact_path"] == "runs/calibration/calibration_summary.json"


def test_vllm_native_segment_requires_score_sequences(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text):
            return []

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(
            LLM=lambda **kwargs: SimpleNamespace(generate=lambda **kwargs: []),
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    gen = VLLMGenerator(model_id="fake-model", scoring_mode="native_segment")

    with pytest.raises(RuntimeError, match="requires a vLLM fork"):
        gen.score_segments([[1]], [[2]], temperature=0.5)


def test_vllm_scoring_mode_is_validated_before_loading_backend():
    with pytest.raises(ValueError, match="unknown vLLM scoring mode"):
        VLLMGenerator(model_id="fake-model", scoring_mode="unsupported")
