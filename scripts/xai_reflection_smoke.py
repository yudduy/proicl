"""xAI GEPA-reflection smoke.

Dry-run mode validates configuration and cost accounting without a network
call. Non-dry-run makes one tiny OpenAI-compatible request through the GEPA LM
adapter and prints only sanitized metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--prompt", default="Return the single word ready.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from polaris.gepa_reflection import (
        XAIReflectionConfig,
        load_env_file,
        make_xai_reflection_lm,
        reflection_manifest,
    )

    load_env_file(args.env_file)
    config = XAIReflectionConfig.from_env(require_key=not args.dry_run)
    estimate = config.estimate_cost(input_tokens=max(1, len(args.prompt) // 4), output_tokens=8)
    if args.dry_run:
        print(
            json.dumps(
                reflection_manifest(
                    provider="xai",
                    config=config,
                    status="dry_run_ready",
                    usage=estimate,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    config.assert_under_caps(estimated_dollar_cost=float(estimate["estimated_dollar_cost"]))
    lm = make_xai_reflection_lm(config)
    response = lm(args.prompt)
    usage = {
        "input_tokens": getattr(lm, "total_tokens_in", 0),
        "output_tokens": getattr(lm, "total_tokens_out", 0),
        "estimated_dollar_cost": getattr(lm, "total_cost", 0.0),
        "response_sha256_length": len(response),
    }
    print(
        json.dumps(
            reflection_manifest(
                provider="xai",
                config=config,
                status="live_request_complete",
                usage=usage,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
