# Vendored from https://github.com/gepa-ai/gepa.git
# @ ce51b50cd196b539c25fae99ad0e0255c23004a4
# Original license: see LICENSE in this directory.
#
# Local modification: trimmed __init__ to only re-export modules that were
# vendored. Upstream additionally re-exports `optimize_anything` and
# `examples.aime`, which polaris does not vendor.
from .adapters import default_adapter
from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .image import Image
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)
