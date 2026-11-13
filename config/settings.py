"""Settings and hardware profiles. Defaults here stay untouched until Phase 7 calibrates them."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILES_DIR = Path(__file__).parent / "profiles"


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    revision: str


class VllmModels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fast: ModelRef
    accurate: ModelRef


class VllmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str
    models: VllmModels
    max_num_seqs: int
    api_servers: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_batched_tokens: int
    cpuset: str


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["direct", "supersample"] = "direct"
    longest_side_px: int = 2048
    workers: int
    cpuset: str
    pdf_cache_size: int = 4
    png_compress_level: int = 1


class CpuOtherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workers: int
    cpuset: str
    nice: int = 10


class MaxTokens(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass1: int = 4096
    pass2: int = 8192


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: MaxTokens = Field(default_factory=MaxTokens)


class Profile(BaseModel):
    """Hardware profile; mirrors the YAML in ARCHITECTURE section 5."""

    model_config = ConfigDict(extra="forbid")
    name: str
    endpoint: str
    vllm: VllmConfig
    render: RenderConfig
    cpu_other: CpuOtherConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    measured: dict[str, float | int | str] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCPARSE_", extra="ignore")

    data_root: Path = Path("data")
    profile: str = "consumer24"

    # QA thresholds (DESIGN section 4). Calibrated in Phase 7 (ADR-027).
    qa_zlib_ratio_min: float = 0.12
    qa_repeat_substring_len: int = 40
    qa_repeat_substring_max: int = 20
    qa_empty_chars_max: int = 20
    qa_empty_text_layer_chars: int = 200
    qa_empty_image_std: float = 8.0
    qa_recall_min: float = 0.40
    qa_precision_min: float = 0.40
    qa_agreement_warn_max: float = 0.75
    qa_token_min_len: int = 2
    qa_letters_only: bool = True  # ADR-025: agreement over alphabetic tokens, maths stripped
    qa_letters_min_len: int = 3


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> Profile:
    path = profiles_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown profile {name!r}: {path} does not exist")
    with path.open(encoding="utf-8") as fh:
        profile = Profile.model_validate(yaml.safe_load(fh))
    endpoint = os.environ.get("DOCPARSE_ENDPOINT")  # ADR-031: point any profile at a remote server
    return profile.model_copy(update={"endpoint": endpoint.rstrip("/")}) if endpoint else profile
