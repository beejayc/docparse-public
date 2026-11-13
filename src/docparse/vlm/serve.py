"""Render compose env from a profile: ``python -m docparse.vlm.serve PROFILE MODEL``."""

from __future__ import annotations

import os
import sys
from typing import Literal

from config.settings import Profile, load_profile


def compose_env(profile: Profile, model: Literal["fast", "accurate"]) -> dict[str, str]:
    v = profile.vllm
    ref = v.models.fast if model == "fast" else v.models.accurate
    return {
        "VLLM_IMAGE": os.environ.get("VLLM_IMAGE") or v.image,
        "MODEL_ID": ref.id,
        "MODEL_REVISION": ref.revision,
        "MAX_NUM_SEQS": str(v.max_num_seqs),
        "GPU_UTIL": str(v.gpu_memory_utilization),
        "API_SERVERS": str(v.api_servers),
        "MAX_MODEL_LEN": str(v.max_model_len),
        "MAX_BATCHED_TOKENS": str(v.max_num_batched_tokens),
        "VLLM_CPUSET": v.cpuset,
    }


def main(argv: list[str]) -> int:
    name, model = argv[1], argv[2]
    if model not in ("fast", "accurate"):
        sys.stderr.write("MODEL must be fast or accurate\n")
        return 2
    env = compose_env(load_profile(name), "fast" if model == "fast" else "accurate")
    sys.stdout.write("".join(f"{k}={v}\n" for k, v in env.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
