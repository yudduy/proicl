from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CloudRiftGPU:
    key: str
    label: str
    vendor: str
    spot_hourly_usd: float
    on_demand_hourly_usd: float
    vram_gb: int
    notes: str
    excluded_by_default: bool = False


@dataclass(frozen=True)
class CloudRiftCostEstimate:
    gpu_key: str
    wall_clock_seconds: float
    hourly_rate_usd: float
    dollars: float
    rate_source: str

    def to_jsonable(self) -> dict:
        return asdict(self)


CLOUDRIFT_GPU_CATALOG: dict[str, CloudRiftGPU] = {
    "rtx4090": CloudRiftGPU(
        key="rtx4090",
        label="RTX 4090",
        vendor="nvidia",
        spot_hourly_usd=0.25,
        on_demand_hourly_usd=0.39,
        vram_gb=24,
        notes="Default first probe target for 1.5B ProRL recovery work.",
    ),
    "v100_sxm3": CloudRiftGPU(
        key="v100_sxm3",
        label="V100 SXM3",
        vendor="nvidia",
        spot_hourly_usd=0.19,
        on_demand_hourly_usd=0.29,
        vram_gb=32,
        notes="Fallback only after FP16 smoke; no native BF16.",
    ),
    "mi350x": CloudRiftGPU(
        key="mi350x",
        label="MI350X",
        vendor="amd",
        spot_hourly_usd=2.37,
        on_demand_hourly_usd=3.65,
        vram_gb=288,
        notes="Excluded unless NVIDIA capacity disappears and ROCm risk is accepted.",
        excluded_by_default=True,
    ),
}


def cloudrift_environment() -> dict[str, str]:
    return {
        "HF_HOME": "/workspace/.cache/huggingface",
        "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
        "POLARIS_REPO_DIR": "/workspace/polaris",
        "POLARIS_RUN_ROOT": "/workspace/polaris/runs/prorl_recovery",
    }


def recommended_gpu_order(*, include_excluded: bool = False) -> tuple[str, ...]:
    keys = ("rtx4090", "v100_sxm3", "mi350x")
    if include_excluded:
        return keys
    return tuple(key for key in keys if not CLOUDRIFT_GPU_CATALOG[key].excluded_by_default)


def estimate_cloudrift_cost(
    gpu_key: str,
    *,
    wall_clock_seconds: float,
    hourly_rate: float | None = None,
    use_spot: bool = True,
) -> CloudRiftCostEstimate:
    if wall_clock_seconds < 0:
        raise ValueError("wall_clock_seconds must be non-negative")
    try:
        gpu = CLOUDRIFT_GPU_CATALOG[gpu_key]
    except KeyError as exc:
        raise ValueError(f"unknown CloudRift GPU: {gpu_key!r}") from exc

    if hourly_rate is None:
        hourly_rate = gpu.spot_hourly_usd if use_spot else gpu.on_demand_hourly_usd
        rate_source = "catalog_spot" if use_spot else "catalog_on_demand"
    else:
        if hourly_rate < 0:
            raise ValueError("hourly_rate must be non-negative")
        rate_source = "explicit_ui_rate"

    dollars = hourly_rate * wall_clock_seconds / 3600.0
    return CloudRiftCostEstimate(
        gpu_key=gpu_key,
        wall_clock_seconds=wall_clock_seconds,
        hourly_rate_usd=hourly_rate,
        dollars=dollars,
        rate_source=rate_source,
    )
