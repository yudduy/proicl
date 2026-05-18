from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from polaris.vendored.gepa.core.adapter import EvaluationBatch


@dataclass(frozen=True)
class PolarisGEPATrace:
    problem_id: str
    prompt_component: str
    prompt_text: str
    generation: str
    score: float
    verifier_result: dict[str, Any]


@dataclass(frozen=True)
class PolarisGEPAOutput:
    problem_id: str
    generation: str
    verifier_result: dict[str, Any]


class PolarisGEPAAdapter:
    """GEPAAdapter-compatible contract for POLARIS prompt co-construction."""

    propose_new_texts = None

    def __init__(
        self,
        *,
        sampler: Any,
        scorer: Callable[[str, str], dict],
        max_new_tokens: int,
        run_dir: Path | None = None,
    ) -> None:
        self.sampler = sampler
        self.scorer = scorer
        self.max_new_tokens = max_new_tokens
        self.run_dir = run_dir
        self.completed_evaluations = 0
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            state_path = run_dir / "adapter_state.json"
            if state_path.exists():
                self.set_adapter_state(json.loads(state_path.read_text(encoding="utf-8")))

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PolarisGEPATrace, PolarisGEPAOutput]:
        component_name, component_text = next(iter(candidate.items()))
        outputs: list[PolarisGEPAOutput] = []
        scores: list[float] = []
        traces: list[PolarisGEPATrace] = []
        prompts = [f"{component_text}{problem.prompt}" for problem in batch]
        batch_generate = getattr(self.sampler, "generate_low_temp_batch", None)
        if callable(batch_generate):
            generations = batch_generate(
                prompts,
                temperature=1.0,
                max_new_tokens=self.max_new_tokens,
            )
        else:
            generations = [
                self.sampler.generate_low_temp(
                    prompt,
                    temperature=1.0,
                    max_new_tokens=self.max_new_tokens,
                )
                for prompt in prompts
            ]
        if len(generations) != len(batch):
            raise RuntimeError(
                "GEPA sampler returned the wrong number of generations: "
                f"expected={len(batch)} observed={len(generations)}"
            )
        for problem, prompt, gen in zip(batch, prompts, generations):
            response = gen.generation if gen.response_contains_prompt else prompt + gen.generation
            verifier_result = self.scorer(response, problem.answer)
            score = float(verifier_result.get("score", 0.0))
            outputs.append(
                PolarisGEPAOutput(
                    problem_id=problem.problem_id,
                    generation=gen.generation,
                    verifier_result=verifier_result,
                )
            )
            scores.append(score)
            if capture_traces:
                traces.append(
                    PolarisGEPATrace(
                        problem_id=problem.problem_id,
                        prompt_component=component_name,
                        prompt_text=prompt,
                        generation=gen.generation,
                        score=score,
                        verifier_result=verifier_result,
                    )
                )
        self.completed_evaluations += len(batch)
        self._persist_state()
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PolarisGEPATrace, PolarisGEPAOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        traces = eval_batch.trajectories or []
        out: dict[str, list[dict[str, Any]]] = {name: [] for name in components_to_update}
        for trace in traces:
            if trace.prompt_component not in out:
                continue
            out[trace.prompt_component].append(
                {
                    "Inputs": {"problem_id": trace.problem_id, "prompt": trace.prompt_text},
                    "Generated Outputs": trace.generation,
                    "Feedback": json.dumps(trace.verifier_result, sort_keys=True),
                    "score": trace.score,
                }
            )
        return out

    def get_adapter_state(self) -> dict[str, Any]:
        return {"completed_evaluations": self.completed_evaluations}

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        self.completed_evaluations = int(state.get("completed_evaluations", 0))

    def _persist_state(self) -> None:
        if self.run_dir is None:
            return
        (self.run_dir / "adapter_state.json").write_text(
            json.dumps(self.get_adapter_state(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
