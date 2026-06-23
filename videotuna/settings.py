"""Centralized PrivTune environment settings (VIDEOTUNA_* prefix)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ComputeBackendSetting = Literal["auto", "cuda", "rocm", "cpu"]
CpuModeSetting = Literal["off", "smoke", "force"]
AttnBackendSetting = Literal["auto", "flash", "sdpa", "eager"]
TorchCompileModeSetting = Literal["reduce-overhead", "max-autotune"]
MetricsOwnerSetting = Literal["script", "flow"]
MetricsBackendSetting = Literal["tensorboard", "trackio"]

ENV_PREFIX = "VIDEOTUNA_"
ENV_COMPUTE_BACKEND = f"{ENV_PREFIX}COMPUTE_BACKEND"
ENV_CPU_MODE = f"{ENV_PREFIX}CPU_MODE"
ENV_ALLOW_CPU_INFERENCE = f"{ENV_PREFIX}ALLOW_CPU_INFERENCE"
ENV_ATTN_BACKEND = f"{ENV_PREFIX}ATTN_BACKEND"
ENV_ATTN_BACKEND_STRICT = f"{ENV_PREFIX}ATTN_BACKEND_STRICT"
ENV_TORCH_COMPILE = f"{ENV_PREFIX}TORCH_COMPILE"
ENV_TORCH_COMPILE_MODE = f"{ENV_PREFIX}TORCH_COMPILE_MODE"
ENV_METRICS_OWNER = f"{ENV_PREFIX}METRICS_OWNER"
ENV_METRICS_BACKEND = f"{ENV_PREFIX}METRICS_BACKEND"
ENV_TRACKIO_SPACE_ID = f"{ENV_PREFIX}TRACKIO_SPACE_ID"
ENV_TRACKIO_PROJECT = f"{ENV_PREFIX}TRACKIO_PROJECT"
ENV_LOG_LEVEL = f"{ENV_PREFIX}LOG_LEVEL"
ENV_BENCH_MODEL = f"{ENV_PREFIX}BENCH_MODEL"

# Non-VIDEOTUNA env vars projected from settings during a session.
# These are saved/restored by settings_session alongside VIDEOTUNA_* vars.
ENV_DIFFUSERS_ATTN_BACKEND = "DIFFUSERS_ATTN_BACKEND"

_VALID_COMPILE_MODES = ("reduce-overhead", "max-autotune")


def _parse_bool01(value: object) -> bool:
    """Parse 0/1 env strings; only ``1`` is truthy (legacy os.environ semantics)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError(f"Expected '0' or '1', got {value!r}")


def _normalize_lower(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower()
    return value


class PrivTuneSettings(BaseSettings):
    """Load all VIDEOTUNA_* environment variables from one settings object."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    compute_backend: ComputeBackendSetting = "auto"
    cpu_mode: CpuModeSetting = "off"
    allow_cpu_inference: bool = False
    attn_backend: AttnBackendSetting = "auto"
    attn_backend_strict: bool = False
    torch_compile: bool = False
    torch_compile_mode: TorchCompileModeSetting = "reduce-overhead"
    metrics_owner: MetricsOwnerSetting = "script"
    metrics_backend: MetricsBackendSetting = "tensorboard"
    trackio_space_id: str | None = None
    trackio_project: str | None = None
    log_level: str = "INFO"
    bench_model: str | None = None

    @field_validator(
        "compute_backend",
        "attn_backend",
        "metrics_owner",
        "metrics_backend",
        mode="before",
    )
    @classmethod
    def _normalize_string_literals(cls, value: object) -> object:
        normalized = _normalize_lower(value)
        if normalized == "mps":
            raise ValueError(
                "VIDEOTUNA_COMPUTE_BACKEND=mps is not supported. "
                "PrivTune supports auto, cuda, rocm, and cpu. "
                "For config validation on Apple Silicon, use "
                "VIDEOTUNA_CPU_MODE=smoke or --cpu-smoke."
            )
        return normalized

    @field_validator("cpu_mode", mode="before")
    @classmethod
    def _normalize_cpu_mode(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip().lower()
            return text if text else "off"
        return value

    @field_validator(
        "allow_cpu_inference",
        "attn_backend_strict",
        "torch_compile",
        mode="before",
    )
    @classmethod
    def _parse_bool01_fields(cls, value: object) -> bool:
        return _parse_bool01(value)

    @field_validator("torch_compile_mode", mode="before")
    @classmethod
    def _normalize_compile_mode(cls, value: object) -> str:
        if value is None:
            return "reduce-overhead"
        mode = str(value).strip()
        if mode in _VALID_COMPILE_MODES:
            return mode
        logger.warning(
            "Invalid {}={!r}; using reduce-overhead",
            ENV_TORCH_COMPILE_MODE,
            mode,
        )
        return "reduce-overhead"

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        return str(value).strip().upper()

    @field_validator(
        "bench_model",
        "trackio_space_id",
        "trackio_project",
        mode="before",
    )
    @classmethod
    def _normalize_optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


_SESSION_SETTINGS: ContextVar[PrivTuneSettings | None] = ContextVar(
    "privtune_session_settings",
    default=None,
)

_SETTINGS_ENV_MAP: dict[str, str] = {
    "compute_backend": ENV_COMPUTE_BACKEND,
    "cpu_mode": ENV_CPU_MODE,
    "allow_cpu_inference": ENV_ALLOW_CPU_INFERENCE,
    "attn_backend": ENV_ATTN_BACKEND,
    "attn_backend_strict": ENV_ATTN_BACKEND_STRICT,
    "torch_compile": ENV_TORCH_COMPILE,
    "torch_compile_mode": ENV_TORCH_COMPILE_MODE,
    "metrics_owner": ENV_METRICS_OWNER,
    "metrics_backend": ENV_METRICS_BACKEND,
    "trackio_space_id": ENV_TRACKIO_SPACE_ID,
    "trackio_project": ENV_TRACKIO_PROJECT,
    "log_level": ENV_LOG_LEVEL,
    "bench_model": ENV_BENCH_MODEL,
}

# Non-VIDEOTUNA env vars that are projected from settings (e.g. diffusers
# internal vars derived from attn_backend). These are saved/restored by
# settings_session so writes within a session don't leak to the parent scope.
_PROJECTED_ENV_KEYS: tuple[str, ...] = (ENV_DIFFUSERS_ATTN_BACKEND,)


def _settings_value_to_env(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def _sync_env_from_settings(settings: PrivTuneSettings) -> dict[str, str | None]:
    """Write session settings to VIDEOTUNA_* env (for subprocess compatibility)."""
    saved: dict[str, str | None] = {}
    payload = settings.model_dump()
    for field_name, env_key in _SETTINGS_ENV_MAP.items():
        saved[env_key] = os.environ.get(env_key)
        os.environ[env_key] = _settings_value_to_env(payload[field_name])
    return saved


def _restore_env(saved: dict[str, str | None]) -> None:
    for env_key, previous in saved.items():
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def get_settings() -> PrivTuneSettings:
    """Return settings loaded from the current environment (no cache)."""
    session = _SESSION_SETTINGS.get()
    if session is not None:
        return session
    return PrivTuneSettings()


@contextmanager
def settings_session(**overrides: Any) -> Iterator[PrivTuneSettings]:
    """Apply session-scoped settings overrides.

    Overrides are visible via :func:`get_settings` and synced to ``VIDEOTUNA_*``
    environment variables for the duration of the context (restored on exit).
    """
    base = _SESSION_SETTINGS.get() or PrivTuneSettings()
    merged = base.model_copy(update=overrides)
    saved_env = _sync_env_from_settings(merged)
    # Save projected (non-VIDEOTUNA) env vars so writes within the session
    # (e.g. DIFFUSERS_ATTN_BACKEND set by apply_diffusers_attention_backend)
    # are restored on exit.
    for key in _PROJECTED_ENV_KEYS:
        saved_env.setdefault(key, os.environ.get(key))
    token = _SESSION_SETTINGS.set(merged)
    try:
        yield merged
    finally:
        _SESSION_SETTINGS.reset(token)
        _restore_env(saved_env)


@contextmanager
def inference_settings_session(
    *,
    cpu_smoke: bool = False,
    compile_flag: bool = False,
) -> Iterator[PrivTuneSettings]:
    """Inference session overrides for CPU smoke and torch.compile."""
    overrides: dict[str, Any] = {}
    if cpu_smoke:
        overrides.update(
            {
                "cpu_mode": "smoke",
                "attn_backend": "eager",
                "torch_compile": False,
            }
        )
    elif compile_flag:
        overrides["torch_compile"] = True
    else:
        overrides["torch_compile"] = False

    with settings_session(**overrides) as settings:
        yield settings
