from __future__ import annotations

from types import SimpleNamespace

from polaris.gepa_integration import PolarisGEPAAdapter


def test_gepa_adapter_uses_sampler_batch_generation_when_available(tmp_path):
    class BatchSampler:
        def __init__(self) -> None:
            self.calls = []

        def generate_low_temp_batch(self, prompts, *, temperature, max_new_tokens):
            self.calls.append(
                {
                    "prompts": list(prompts),
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return [
                SimpleNamespace(
                    generation=f" answer-{idx}",
                    response_contains_prompt=False,
                )
                for idx, _ in enumerate(prompts)
            ]

    sampler = BatchSampler()
    adapter = PolarisGEPAAdapter(
        sampler=sampler,
        scorer=lambda generation, answer: {
            "score": 1.0 if answer in generation else 0.0,
            "passed": answer in generation,
        },
        max_new_tokens=32,
        run_dir=tmp_path,
    )
    batch = [
        SimpleNamespace(problem_id="p0", prompt="Prompt 0", answer="answer-0"),
        SimpleNamespace(problem_id="p1", prompt="Prompt 1", answer="answer-1"),
    ]

    result = adapter.evaluate(
        batch,
        {"instruction": "Instruction\n"},
        capture_traces=True,
    )

    assert len(sampler.calls) == 1
    assert sampler.calls[0]["prompts"] == [
        "Instruction\nPrompt 0",
        "Instruction\nPrompt 1",
    ]
    assert sampler.calls[0]["temperature"] == 1.0
    assert sampler.calls[0]["max_new_tokens"] == 32
    assert result.scores == [1.0, 1.0]
    assert result.num_metric_calls == 2
