from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, MutableMapping


XAI_BASE_URL_DEFAULT = "https://api.x.ai/v1"
XAI_MODEL_DEFAULT = "grok-4.3"
XAI_LITELLM_MODEL_DEFAULT = f"openai/{XAI_MODEL_DEFAULT}"
XAI_INPUT_PRICE_PER_MILLION = 1.25
XAI_OUTPUT_PRICE_PER_MILLION = 2.50
XAI_INITIAL_REFLECTION_CAP_DOLLARS = 30.0
XAI_HARD_REFLECTION_CAP_DOLLARS = 100.0
LOCAL_HF_REFLECTION_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
REDACTED = "<redacted>"


def load_env_file(path: Path, *, env: MutableMapping[str, str] | None = None) -> None:
    target = env if env is not None else os.environ
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in target:
            target[key] = value


def estimate_xai_reflection_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float = XAI_INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = XAI_OUTPUT_PRICE_PER_MILLION,
) -> dict[str, float | int]:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    cost = (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "estimated_dollar_cost": cost,
    }


@dataclass(frozen=True)
class XAIReflectionConfig:
    api_key: str | None
    base_url: str = XAI_BASE_URL_DEFAULT
    model: str = XAI_MODEL_DEFAULT
    litellm_model: str = XAI_LITELLM_MODEL_DEFAULT
    input_price_per_million: float = XAI_INPUT_PRICE_PER_MILLION
    output_price_per_million: float = XAI_OUTPUT_PRICE_PER_MILLION
    initial_cost_cap_dollars: float = XAI_INITIAL_REFLECTION_CAP_DOLLARS
    hard_cost_cap_dollars: float = XAI_HARD_REFLECTION_CAP_DOLLARS
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def from_env(
        cls,
        env: MutableMapping[str, str] | None = None,
        *,
        require_key: bool = True,
    ) -> "XAIReflectionConfig":
        source = env if env is not None else os.environ
        api_key = source.get("XAI_API_KEY")
        if require_key and not api_key:
            raise ValueError("XAI_API_KEY is required for xAI reflection")
        model = source.get("XAI_REFLECTION_MODEL", XAI_MODEL_DEFAULT)
        return cls(
            api_key=api_key,
            base_url=source.get("XAI_BASE_URL", XAI_BASE_URL_DEFAULT),
            model=model,
            litellm_model=source.get("XAI_LITELLM_MODEL", f"openai/{model}"),
            input_price_per_million=float(
                source.get("XAI_INPUT_PRICE_PER_MILLION", XAI_INPUT_PRICE_PER_MILLION)
            ),
            output_price_per_million=float(
                source.get("XAI_OUTPUT_PRICE_PER_MILLION", XAI_OUTPUT_PRICE_PER_MILLION)
            ),
            initial_cost_cap_dollars=float(
                source.get(
                    "XAI_REFLECTION_INITIAL_CAP_DOLLARS",
                    XAI_INITIAL_REFLECTION_CAP_DOLLARS,
                )
            ),
            hard_cost_cap_dollars=float(
                source.get(
                    "XAI_REFLECTION_HARD_CAP_DOLLARS",
                    XAI_HARD_REFLECTION_CAP_DOLLARS,
                )
            ),
            temperature=float(source.get("XAI_REFLECTION_TEMPERATURE", 0.7)),
            max_tokens=int(source.get("XAI_REFLECTION_MAX_TOKENS", 1024)),
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["api_key"] = REDACTED if self.api_key else None
        return payload

    def estimate_cost(self, *, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        estimate = estimate_xai_reflection_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price_per_million=self.input_price_per_million,
            output_price_per_million=self.output_price_per_million,
        )
        estimate["under_initial_cap"] = (
            float(estimate["estimated_dollar_cost"]) <= self.initial_cost_cap_dollars
        )
        estimate["under_hard_cap"] = (
            float(estimate["estimated_dollar_cost"]) <= self.hard_cost_cap_dollars
        )
        return estimate

    def assert_under_caps(self, *, estimated_dollar_cost: float) -> None:
        if estimated_dollar_cost > self.initial_cost_cap_dollars:
            raise ValueError(
                "xAI reflection estimate exceeds initial cap "
                f"({estimated_dollar_cost} > {self.initial_cost_cap_dollars})"
            )
        if estimated_dollar_cost > self.hard_cost_cap_dollars:
            raise ValueError(
                "xAI reflection estimate exceeds hard cap "
                f"({estimated_dollar_cost} > {self.hard_cost_cap_dollars})"
            )


@dataclass(frozen=True)
class LocalHFReflectionConfig:
    model_id: str = LOCAL_HF_REFLECTION_MODEL_DEFAULT
    revision: str | None = None
    temperature: float = 0.7
    max_new_tokens: int = 1024
    local_files_only: bool = False
    device: str = "cuda"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


class LocalHFReflectionLM:
    """Local Hugging Face reflection model for GEPA without paid API calls."""

    def __init__(self, config: LocalHFReflectionConfig) -> None:
        import torch
        import transformers

        self.config = config
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._total_cost = 0.0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if callable(getattr(self.tokenizer, "apply_chat_template", None)):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = str(prompt)
        input_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.model.device)
        started_tokens = int(input_ids["input_ids"].shape[-1])
        with self.torch.no_grad():
            output = self.model.generate(
                **input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][started_tokens:].detach().to("cpu")
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self._total_tokens_in += started_tokens
        self._total_tokens_out += int(generated.numel())
        return text


class CapEnforcedReflectionLM:
    """Wrap a paid reflection LM and fail closed before crossing configured caps."""

    _CHARS_PER_TOKEN = 4

    def __init__(self, lm: Any, config: XAIReflectionConfig) -> None:
        self._lm = lm
        self._config = config

    @property
    def total_cost(self) -> float:
        return float(getattr(self._lm, "total_cost", 0.0) or 0.0)

    @property
    def total_tokens_in(self) -> int:
        return int(getattr(self._lm, "total_tokens_in", 0) or 0)

    @property
    def total_tokens_out(self) -> int:
        return int(getattr(self._lm, "total_tokens_out", 0) or 0)

    def _estimate_prompt_tokens(self, prompt: str | list[dict[str, Any]]) -> int:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def _assert_projected_call_under_caps(self, prompt: str | list[dict[str, Any]]) -> None:
        estimate = self._config.estimate_cost(
            input_tokens=self.total_tokens_in + self._estimate_prompt_tokens(prompt),
            output_tokens=self.total_tokens_out + self._config.max_tokens,
        )
        self._config.assert_under_caps(
            estimated_dollar_cost=float(estimate["estimated_dollar_cost"])
        )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self._assert_projected_call_under_caps(prompt)
        result = self._lm(prompt)
        self._config.assert_under_caps(estimated_dollar_cost=self.total_cost)
        return result


def make_xai_reflection_lm(config: XAIReflectionConfig):
    if not config.api_key:
        raise ValueError("XAI_API_KEY is required for xAI reflection")
    try:
        from polaris.vendored.gepa.lm import LM
    except ModuleNotFoundError as exc:
        if exc.name == "litellm":
            raise RuntimeError(
                "live xAI reflection requires the gepa_reflection extra: "
                "python -m pip install -e '.[gepa_reflection]'"
            ) from exc
        raise

    lm = LM(
        config.litellm_model,
        api_key=config.api_key,
        api_base=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return CapEnforcedReflectionLM(lm, config)


def make_local_hf_reflection_lm(config: LocalHFReflectionConfig) -> LocalHFReflectionLM:
    return LocalHFReflectionLM(config)


def reflection_manifest(
    *,
    provider: str,
    config: XAIReflectionConfig | LocalHFReflectionConfig | None,
    status: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "provider": provider,
        "status": status,
        "usage": usage or {},
    }
    if config is not None:
        payload["config"] = config.to_manifest()
    serialized = json.dumps(payload)
    api_key = getattr(config, "api_key", None)
    if api_key and api_key in serialized:
        raise RuntimeError("secret leaked into reflection manifest")
    return payload
