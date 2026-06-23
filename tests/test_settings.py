"""Tests for videotuna.settings module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from pydantic_settings import SettingsConfigDict

from videotuna.settings import (
    _SETTINGS_ENV_MAP,
    ENV_ALLOW_CPU_INFERENCE,
    ENV_ATTN_BACKEND,
    ENV_ATTN_BACKEND_STRICT,
    ENV_BENCH_MODEL,
    ENV_COMPUTE_BACKEND,
    ENV_CPU_MODE,
    ENV_LOG_LEVEL,
    ENV_PREFIX,
    ENV_TORCH_COMPILE,
    PrivTuneSettings,
    _parse_bool01,
    _restore_env,
    _settings_value_to_env,
    _sync_env_from_settings,
    get_settings,
    settings_session,
)


class TestParseBool01:
    """Test the _parse_bool01 boolean parser."""

    def test_parse_bool01_accepts_bool_true(self):
        assert _parse_bool01(True) is True

    def test_parse_bool01_accepts_bool_false(self):
        assert _parse_bool01(False) is False

    def test_parse_bool01_accepts_string_1(self):
        assert _parse_bool01("1") is True

    def test_parse_bool01_accepts_string_0(self):
        assert _parse_bool01("0") is False

    def test_parse_bool01_accepts_string_1_with_whitespace(self):
        assert _parse_bool01("  1  ") is True

    def test_parse_bool01_accepts_string_0_with_whitespace(self):
        assert _parse_bool01("  0  ") is False

    def test_parse_bool01_accepts_none_as_false(self):
        assert _parse_bool01(None) is False

    def test_parse_bool01_rejects_true_string(self):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01("true")

    def test_parse_bool01_rejects_false_string(self):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01("false")

    def test_parse_bool01_rejects_yes(self):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01("yes")

    def test_parse_bool01_rejects_no(self):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01("no")

    def test_parse_bool01_accepts_int_1(self):
        assert _parse_bool01(1) is True

    def test_parse_bool01_accepts_int_0(self):
        assert _parse_bool01(0) is False

    @pytest.mark.parametrize(
        "bad_value",
        ["", "2", "01", "10", "True", "False", "on", "off", "maybe", "  "],
    )
    def test_parse_bool01_rejects_non_bool01_strings(self, bad_value):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01(bad_value)

    def test_parse_bool01_rejects_int_2(self):
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            _parse_bool01(2)


class TestPrivTuneSettingsDefaults:
    """Test PrivTuneSettings defaults and basic initialization."""

    def test_default_values(self):
        """Verify all field defaults match specification."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_COMPUTE_BACKEND, None)
            os.environ.pop(ENV_CPU_MODE, None)
            os.environ.pop(ENV_ALLOW_CPU_INFERENCE, None)
            os.environ.pop(ENV_ATTN_BACKEND, None)
            os.environ.pop(ENV_ATTN_BACKEND_STRICT, None)
            os.environ.pop(ENV_TORCH_COMPILE, None)

            settings = PrivTuneSettings()
            assert settings.compute_backend == "auto"
            assert settings.cpu_mode == "off"
            assert settings.allow_cpu_inference is False
            assert settings.attn_backend == "auto"
            assert settings.attn_backend_strict is False
            assert settings.torch_compile is False
            assert settings.torch_compile_mode == "reduce-overhead"
            assert settings.metrics_owner == "script"
            assert settings.metrics_backend == "tensorboard"
            assert settings.trackio_space_id is None
            assert settings.trackio_project is None
            assert settings.log_level == "INFO"
            assert settings.bench_model is None


class TestBooleanFieldParsing:
    """Test boolean field parsing for allow_cpu_inference, attn_backend_strict."""

    def test_allow_cpu_inference_from_string_1(self):
        """VIDEOTUNA_ALLOW_CPU_INFERENCE=1 should be True."""
        settings = PrivTuneSettings(allow_cpu_inference="1")
        assert settings.allow_cpu_inference is True

    def test_allow_cpu_inference_from_string_0(self):
        """VIDEOTUNA_ALLOW_CPU_INFERENCE=0 should be False."""
        settings = PrivTuneSettings(allow_cpu_inference="0")
        assert settings.allow_cpu_inference is False

    def test_allow_cpu_inference_rejects_true_string(self):
        """VIDEOTUNA_ALLOW_CPU_INFERENCE=true should raise ValueError."""
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            PrivTuneSettings(allow_cpu_inference="true")

    def test_allow_cpu_inference_rejects_false_string(self):
        """VIDEOTUNA_ALLOW_CPU_INFERENCE=false should raise ValueError."""
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            PrivTuneSettings(allow_cpu_inference="false")

    def test_attn_backend_strict_from_string_1(self):
        """VIDEOTUNA_ATTN_BACKEND_STRICT=1 should be True."""
        settings = PrivTuneSettings(attn_backend_strict="1")
        assert settings.attn_backend_strict is True

    def test_attn_backend_strict_from_string_0(self):
        """VIDEOTUNA_ATTN_BACKEND_STRICT=0 should be False."""
        settings = PrivTuneSettings(attn_backend_strict="0")
        assert settings.attn_backend_strict is False

    def test_attn_backend_strict_rejects_yes(self):
        """VIDEOTUNA_ATTN_BACKEND_STRICT=yes should raise ValueError."""
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            PrivTuneSettings(attn_backend_strict="yes")

    def test_torch_compile_from_string_1(self):
        """VIDEOTUNA_TORCH_COMPILE=1 should be True."""
        settings = PrivTuneSettings(torch_compile="1")
        assert settings.torch_compile is True

    def test_torch_compile_from_string_0(self):
        """VIDEOTUNA_TORCH_COMPILE=0 should be False."""
        settings = PrivTuneSettings(torch_compile="0")
        assert settings.torch_compile is False

    def test_torch_compile_rejects_no(self):
        """VIDEOTUNA_TORCH_COMPILE=no should raise ValueError."""
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            PrivTuneSettings(torch_compile="no")

    @pytest.mark.parametrize(
        "field_name",
        ["allow_cpu_inference", "attn_backend_strict", "torch_compile"],
    )
    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("1", True),
            ("0", False),
            (1, True),
            (0, False),
            (True, True),
            (False, False),
        ],
    )
    def test_boolean_fields_delegate_to_parse_bool01(
        self, field_name, input_value, expected
    ):
        """Boolean fields should normalize through _parse_bool01."""
        settings = PrivTuneSettings(**{field_name: input_value})
        assert getattr(settings, field_name) is expected

    @pytest.mark.parametrize(
        "field_name",
        ["allow_cpu_inference", "attn_backend_strict", "torch_compile"],
    )
    @pytest.mark.parametrize(
        "bad_value",
        ["true", "false", "yes", "no", "on", "off", "", "2", "01"],
    )
    def test_boolean_fields_reject_non_bool01_values(self, field_name, bad_value):
        """Boolean fields should reject non 0/1 values."""
        with pytest.raises(ValueError, match="Expected '0' or '1'"):
            PrivTuneSettings(**{field_name: bad_value})


class TestStringFieldNormalization:
    """Test normalization of string-like enum fields."""

    def test_compute_backend_lowercase_normalization(self):
        """compute_backend should normalize to lowercase."""
        settings = PrivTuneSettings(compute_backend="AUTO")
        assert settings.compute_backend == "auto"

    def test_compute_backend_with_whitespace(self):
        """compute_backend should strip whitespace."""
        settings = PrivTuneSettings(compute_backend="  CUDA  ")
        assert settings.compute_backend == "cuda"

    def test_attn_backend_lowercase_normalization(self):
        """attn_backend should normalize to lowercase."""
        settings = PrivTuneSettings(attn_backend="FLASH")
        assert settings.attn_backend == "flash"

    def test_attn_backend_with_whitespace(self):
        """attn_backend should strip whitespace."""
        settings = PrivTuneSettings(attn_backend="  SDPA  ")
        assert settings.attn_backend == "sdpa"

    def test_metrics_owner_lowercase_normalization(self):
        """metrics_owner should normalize to lowercase."""
        settings = PrivTuneSettings(metrics_owner="FLOW")
        assert settings.metrics_owner == "flow"

    def test_metrics_backend_lowercase_normalization(self):
        """metrics_backend should normalize to lowercase."""
        settings = PrivTuneSettings(metrics_backend="TRACKIO")
        assert settings.metrics_backend == "trackio"

    def test_cpu_mode_lowercase_normalization(self):
        """cpu_mode should normalize to lowercase."""
        settings = PrivTuneSettings(cpu_mode="SMOKE")
        assert settings.cpu_mode == "smoke"

    def test_cpu_mode_empty_string_defaults_to_off(self):
        """cpu_mode with empty string should default to 'off'."""
        settings = PrivTuneSettings(cpu_mode="")
        assert settings.cpu_mode == "off"

    def test_log_level_uppercase_normalization(self):
        """log_level should normalize to uppercase."""
        settings = PrivTuneSettings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_log_level_with_whitespace(self):
        """log_level should strip whitespace."""
        settings = PrivTuneSettings(log_level="  warning  ")
        assert settings.log_level == "WARNING"


class TestOptionalStringFieldNormalization:
    """Test normalization of optional string fields."""

    def test_bench_model_none_stays_none(self):
        """bench_model=None should stay None."""
        settings = PrivTuneSettings(bench_model=None)
        assert settings.bench_model is None

    def test_bench_model_empty_string_becomes_none(self):
        """bench_model with empty string should become None."""
        settings = PrivTuneSettings(bench_model="")
        assert settings.bench_model is None

    def test_bench_model_strips_whitespace(self):
        """bench_model should strip whitespace."""
        settings = PrivTuneSettings(bench_model="  model-name  ")
        assert settings.bench_model == "model-name"

    def test_trackio_space_id_whitespace_becomes_none(self):
        """trackio_space_id with only whitespace should become None."""
        settings = PrivTuneSettings(trackio_space_id="   ")
        assert settings.trackio_space_id is None

    def test_trackio_project_empty_becomes_none(self):
        """trackio_project empty string should become None."""
        settings = PrivTuneSettings(trackio_project="")
        assert settings.trackio_project is None


class TestInvalidEnumValues:
    """Test that invalid enum-like values are rejected."""

    def test_compute_backend_mps_explicitly_rejected(self):
        """VIDEOTUNA_COMPUTE_BACKEND=mps should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="VIDEOTUNA_COMPUTE_BACKEND=mps is not supported",
        ):
            PrivTuneSettings(compute_backend="mps")

    def test_compute_backend_invalid_value(self):
        """compute_backend with invalid value should raise validation error."""
        with pytest.raises(
            ValueError,
            match="Input should be 'auto', 'cuda', 'rocm' or 'cpu'",
        ):
            PrivTuneSettings(compute_backend="invalid_backend")

    def test_attn_backend_invalid_value(self):
        """attn_backend with invalid value should raise validation error."""
        with pytest.raises(
            ValueError,
            match="Input should be 'auto', 'flash', 'sdpa' or 'eager'",
        ):
            PrivTuneSettings(attn_backend="invalid_attn")

    def test_metrics_owner_invalid_value(self):
        """metrics_owner with invalid value should raise validation error."""
        with pytest.raises(
            ValueError,
            match="Input should be 'script' or 'flow'",
        ):
            PrivTuneSettings(metrics_owner="invalid_owner")

    def test_metrics_backend_invalid_value(self):
        """metrics_backend with invalid value should raise validation error."""
        with pytest.raises(
            ValueError,
            match="Input should be 'tensorboard' or 'trackio'",
        ):
            PrivTuneSettings(metrics_backend="invalid_backend")

    def test_cpu_mode_invalid_value(self):
        """cpu_mode with invalid value should raise validation error."""
        with pytest.raises(
            ValueError,
            match="Input should be 'off', 'smoke' or 'force'",
        ):
            PrivTuneSettings(cpu_mode="invalid_mode")

    @pytest.mark.parametrize(
        "field_name,bad_value,expected_pattern",
        [
            (
                "compute_backend",
                "tpu",
                "Input should be 'auto', 'cuda', 'rocm' or 'cpu'",
            ),
            (
                "attn_backend",
                "xformers",
                "Input should be 'auto', 'flash', 'sdpa' or 'eager'",
            ),
            (
                "cpu_mode",
                "fast",
                "Input should be 'off', 'smoke' or 'force'",
            ),
            (
                "metrics_owner",
                "service",
                "Input should be 'script' or 'flow'",
            ),
            (
                "metrics_backend",
                "wandb",
                "Input should be 'tensorboard' or 'trackio'",
            ),
        ],
    )
    def test_env_driven_invalid_enum_values(
        self, field_name, bad_value, expected_pattern
    ):
        """Invalid env values should be rejected during construction."""
        env_key = _SETTINGS_ENV_MAP[field_name]
        with mock.patch.dict(os.environ, {env_key: bad_value}, clear=False):
            with pytest.raises(ValueError, match=expected_pattern):
                PrivTuneSettings()

    def test_env_driven_mps_rejected(self):
        """VIDEOTUNA_COMPUTE_BACKEND=mps via env should be rejected."""
        with mock.patch.dict(os.environ, {ENV_COMPUTE_BACKEND: "mps"}, clear=False):
            with pytest.raises(
                ValueError, match="VIDEOTUNA_COMPUTE_BACKEND=mps is not supported"
            ):
                PrivTuneSettings()

    @pytest.mark.parametrize(
        "env_key",
        [ENV_ALLOW_CPU_INFERENCE, ENV_ATTN_BACKEND_STRICT, ENV_TORCH_COMPILE],
    )
    def test_env_driven_invalid_boolean_values(self, env_key):
        """Non-0/1 boolean env values should be rejected."""
        with mock.patch.dict(os.environ, {env_key: "true"}, clear=False):
            with pytest.raises(ValueError, match="Expected '0' or '1'"):
                PrivTuneSettings()


class TestEnvFileLoading:
    """Test .env file loading with case-insensitive names."""

    def test_env_file_loading_basic(self):
        """Settings should load from .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("VIDEOTUNA_COMPUTE_BACKEND=cuda\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(ENV_COMPUTE_BACKEND, None)
                settings = PrivTuneSettings(_env_file=str(env_file))
                assert settings.compute_backend == "cuda"

    def test_case_insensitive_env_vars(self):
        """Settings should accept case-insensitive environment variable names."""
        with mock.patch.dict(
            os.environ,
            {
                "videotuna_compute_backend": "rocm",
                "VIDEOTUNA_ATTN_BACKEND": "SDPA",
                "VideoTuna_Torch_Compile": "1",
            },
            clear=False,
        ):
            settings = PrivTuneSettings()
            assert settings.compute_backend == "rocm"
            assert settings.attn_backend == "sdpa"
            assert settings.torch_compile is True

    def test_env_file_with_mixed_case_keys(self):
        """Settings should load from .env file with mixed-case keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "videotuna_compute_backend=cpu\nVIDEOTUNA_LOG_LEVEL=debug\n"
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(ENV_COMPUTE_BACKEND, None)
                os.environ.pop(ENV_LOG_LEVEL, None)
                settings = PrivTuneSettings(_env_file=str(env_file))
                assert settings.compute_backend == "cpu"
                assert settings.log_level == "DEBUG"

    def test_env_file_all_supported_variables(self):
        """Settings should load every supported VIDEOTUNA_* variable from .env."""
        env_content = (
            "VIDEOTUNA_COMPUTE_BACKEND=rocm\n"
            "VIDEOTUNA_CPU_MODE=smoke\n"
            "VIDEOTUNA_ALLOW_CPU_INFERENCE=1\n"
            "VIDEOTUNA_ATTN_BACKEND=SDPA\n"
            "VIDEOTUNA_ATTN_BACKEND_STRICT=0\n"
            "VIDEOTUNA_TORCH_COMPILE=1\n"
            "VIDEOTUNA_TORCH_COMPILE_MODE=max-autotune\n"
            "VIDEOTUNA_METRICS_OWNER=FLOW\n"
            "VIDEOTUNA_METRICS_BACKEND=TRACKIO\n"
            "VIDEOTUNA_TRACKIO_SPACE_ID=space-123\n"
            "VIDEOTUNA_TRACKIO_PROJECT=project-abc\n"
            "VIDEOTUNA_LOG_LEVEL=debug\n"
            "VIDEOTUNA_BENCH_MODEL=bench-model\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(env_content)
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in _SETTINGS_ENV_MAP.values():
                    os.environ.pop(key, None)
                settings = PrivTuneSettings(_env_file=str(env_file))
                assert settings.compute_backend == "rocm"
                assert settings.cpu_mode == "smoke"
                assert settings.allow_cpu_inference is True
                assert settings.attn_backend == "sdpa"
                assert settings.attn_backend_strict is False
                assert settings.torch_compile is True
                assert settings.torch_compile_mode == "max-autotune"
                assert settings.metrics_owner == "flow"
                assert settings.metrics_backend == "trackio"
                assert settings.trackio_space_id == "space-123"
                assert settings.trackio_project == "project-abc"
                assert settings.log_level == "DEBUG"
                assert settings.bench_model == "bench-model"

    def test_env_file_mixed_case_keys_all_fields(self):
        """.env file keys should be case-insensitive."""
        env_content = (
            "videotuna_compute_backend=cpu\n"
            "VideoTuna_Cpu_Mode=force\n"
            "VIDEOtuna_ALLOW_CPU_INFERENCE=1\n"
            "VIDEOTUNA_ATTN_BACKEND=EAGER\n"
            "videotuna_attn_backend_strict=1\n"
            "VideoTuna_Torch_Compile=0\n"
            "VIDEOTUNA_TORCH_COMPILE_MODE=REDUCE-OVERHEAD\n"
            "videotuna_metrics_owner=flow\n"
            "VIDEOTUNA_METRICS_BACKEND=tensorboard\n"
            "VideoTuna_Trackio_Space_Id=space-456\n"
            "videotuna_trackio_project=project-def\n"
            "VIDEOTUNA_LOG_LEVEL=warning\n"
            "videotuna_bench_model=\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(env_content)
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in _SETTINGS_ENV_MAP.values():
                    os.environ.pop(key, None)
                settings = PrivTuneSettings(_env_file=str(env_file))
                assert settings.compute_backend == "cpu"
                assert settings.cpu_mode == "force"
                assert settings.allow_cpu_inference is True
                assert settings.attn_backend == "eager"
                assert settings.attn_backend_strict is True
                assert settings.torch_compile is False
                assert settings.torch_compile_mode == "reduce-overhead"
                assert settings.metrics_owner == "flow"
                assert settings.metrics_backend == "tensorboard"
                assert settings.trackio_space_id == "space-456"
                assert settings.trackio_project == "project-def"
                assert settings.log_level == "WARNING"
                assert settings.bench_model is None

    def test_env_file_via_model_config_subclass(self):
        """Env file can be configured via a subclass with model_config.env_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("VIDEOTUNA_COMPUTE_BACKEND=cpu\n")

            class LocalEnvSettings(PrivTuneSettings):
                model_config = SettingsConfigDict(
                    env_prefix=ENV_PREFIX,
                    env_file=str(env_file),
                    env_file_encoding="utf-8",
                    extra="ignore",
                    case_sensitive=False,
                )

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(ENV_COMPUTE_BACKEND, None)
                settings = LocalEnvSettings()
                assert settings.compute_backend == "cpu"

    def test_env_file_precedence_env_over_dotenv(self):
        """Environment variables should override .env file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "VIDEOTUNA_COMPUTE_BACKEND=cuda\n"
                "VIDEOTUNA_TORCH_COMPILE=1\n"
                "VIDEOTUNA_BENCH_MODEL=from-env-file\n"
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in _SETTINGS_ENV_MAP.values():
                    os.environ.pop(key, None)
                with mock.patch.dict(
                    os.environ,
                    {
                        ENV_COMPUTE_BACKEND: "cpu",
                        ENV_BENCH_MODEL: "from-env-var",
                    },
                    clear=False,
                ):
                    settings = PrivTuneSettings(_env_file=str(env_file))
                    assert settings.compute_backend == "cpu"  # env wins
                    assert settings.torch_compile is True  # from .env
                    assert settings.bench_model == "from-env-var"  # env wins

    def test_env_file_precedence_constructor_over_dotenv(self):
        """Constructor arguments should override .env file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "VIDEOTUNA_COMPUTE_BACKEND=cuda\n"
                "VIDEOTUNA_TORCH_COMPILE=1\n"
                "VIDEOTUNA_BENCH_MODEL=from-env-file\n"
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in _SETTINGS_ENV_MAP.values():
                    os.environ.pop(key, None)
                settings = PrivTuneSettings(
                    _env_file=str(env_file),
                    compute_backend="rocm",
                    torch_compile=False,
                    bench_model=None,
                )
                assert settings.compute_backend == "rocm"  # constructor wins
                assert settings.torch_compile is False  # constructor wins
                assert settings.bench_model is None  # constructor wins

    def test_env_file_normalizes_all_field_types(self):
        """.env values normalize across string, boolean, and optional-string fields."""
        env_content = (
            "VIDEOTUNA_COMPUTE_BACKEND= CUDA \n"
            "VIDEOTUNA_CPU_MODE= SMOKE \n"
            "VIDEOTUNA_ALLOW_CPU_INFERENCE=1\n"
            "VIDEOTUNA_ATTN_BACKEND= EAGER \n"
            "VIDEOTUNA_ATTN_BACKEND_STRICT=0\n"
            "VIDEOTUNA_TORCH_COMPILE=1\n"
            "VIDEOTUNA_LOG_LEVEL= debug \n"
            "VIDEOTUNA_BENCH_MODEL=   \n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(env_content)
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in _SETTINGS_ENV_MAP.values():
                    os.environ.pop(key, None)
                settings = PrivTuneSettings(_env_file=str(env_file))
                assert settings.compute_backend == "cuda"  # string normalization
                assert settings.cpu_mode == "smoke"  # string normalization
                assert settings.allow_cpu_inference is True  # boolean normalization
                assert settings.attn_backend == "eager"  # string normalization
                assert settings.attn_backend_strict is False  # boolean normalization
                assert settings.torch_compile is True  # boolean normalization
                assert settings.log_level == "DEBUG"  # string normalization
                assert settings.bench_model is None  # optional-string normalization


class TestGetSettings:
    """Test get_settings() function behavior."""

    def test_get_settings_no_caching_outside_session(self):
        """get_settings() should not cache globally outside a session."""
        with mock.patch.dict(os.environ, {ENV_COMPUTE_BACKEND: "cuda"}, clear=True):
            settings1 = get_settings()
            assert settings1.compute_backend == "cuda"

        with mock.patch.dict(os.environ, {ENV_COMPUTE_BACKEND: "rocm"}, clear=True):
            settings2 = get_settings()
            assert settings2.compute_backend == "rocm"

    def test_get_settings_returns_new_instance_each_time(self):
        """get_settings() outside a session should return different instances."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_get_settings_inside_session_returns_session_object(self):
        """get_settings() inside settings_session should return the session object."""
        with settings_session(compute_backend="cuda") as session:
            settings = get_settings()
            assert settings is session
            assert settings.compute_backend == "cuda"

    def test_get_settings_outside_session_after_session(self):
        """get_settings() outside a session should not be affected by prior session."""
        with settings_session(compute_backend="cuda"):
            pass

        with mock.patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
            assert settings.compute_backend == "auto"

    def test_get_settings_reflects_environment_changes(self):
        """get_settings() reflects environment changes when no session is active."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ[ENV_COMPUTE_BACKEND] = "cuda"
            assert get_settings().compute_backend == "cuda"

            os.environ[ENV_COMPUTE_BACKEND] = "cpu"
            assert get_settings().compute_backend == "cpu"

            os.environ[ENV_TORCH_COMPILE] = "1"
            assert get_settings().torch_compile is True

            os.environ[ENV_TORCH_COMPILE] = "0"
            assert get_settings().torch_compile is False

    def test_get_settings_session_instance_identity(self):
        """get_settings() should return the exact session object repeatedly."""
        with settings_session(compute_backend="cuda") as session:
            assert get_settings() is session
            assert get_settings() is session
            assert get_settings().compute_backend == "cuda"


class TestSettingsSession:
    """Test settings_session() context manager."""

    def test_settings_session_applies_overrides(self):
        """settings_session should apply overrides to settings."""
        with settings_session(compute_backend="cuda", torch_compile=True) as s:
            assert s.compute_backend == "cuda"
            assert s.torch_compile is True

    def test_settings_session_syncs_to_env(self):
        """settings_session should sync settings to VIDEOTUNA_* env vars."""
        with settings_session(compute_backend="rocm", log_level="debug"):
            assert os.environ.get(ENV_COMPUTE_BACKEND) == "rocm"
            assert os.environ.get(ENV_LOG_LEVEL) == "debug"

    def test_settings_session_restores_env_on_exit(self):
        """settings_session should restore original env vars on exit."""
        original_value = os.environ.get(ENV_COMPUTE_BACKEND)
        os.environ[ENV_COMPUTE_BACKEND] = "cuda"

        try:
            with settings_session(compute_backend="rocm"):
                assert os.environ[ENV_COMPUTE_BACKEND] == "rocm"

            assert os.environ.get(ENV_COMPUTE_BACKEND) == "cuda"
        finally:
            if original_value is None:
                os.environ.pop(ENV_COMPUTE_BACKEND, None)
            else:
                os.environ[ENV_COMPUTE_BACKEND] = original_value

    def test_settings_session_restores_unset_env_vars(self):
        """settings_session should remove env vars that were not set before."""
        os.environ.pop(ENV_ATTN_BACKEND, None)

        with settings_session(attn_backend="flash"):
            assert os.environ.get(ENV_ATTN_BACKEND) == "flash"

        assert os.environ.get(ENV_ATTN_BACKEND) is None

    def test_settings_session_nested_context(self):
        """Nested settings_session should work correctly."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_COMPUTE_BACKEND, None)

            with settings_session(compute_backend="cuda") as s1:
                assert s1.compute_backend == "cuda"

                with settings_session(compute_backend="rocm") as s2:
                    assert s2.compute_backend == "rocm"
                    assert get_settings().compute_backend == "rocm"

                assert get_settings().compute_backend == "cuda"

    def test_settings_session_preserves_unspecified_values(self):
        """settings_session overrides should preserve unspecified values from base."""
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ.keys()):
                if key.startswith(ENV_PREFIX):
                    os.environ.pop(key)

            with settings_session(
                compute_backend="cuda", log_level="debug", torch_compile=True
            ) as base:
                assert base.compute_backend == "cuda"
                assert base.log_level == "debug"
                assert base.torch_compile is True

                with settings_session(log_level="info") as merged:
                    assert merged.compute_backend == "cuda"
                    assert merged.log_level == "info"
                    assert merged.torch_compile is True

    def test_settings_session_env_sync_boolean_values(self):
        """settings_session should sync boolean values as '0'/'1'."""
        with settings_session(torch_compile=True, allow_cpu_inference=False):
            assert os.environ.get(ENV_TORCH_COMPILE) == "1"
            assert os.environ.get(ENV_ALLOW_CPU_INFERENCE) == "0"

    def test_settings_session_env_sync_none_values(self):
        """settings_session should clear env vars for None values."""
        with mock.patch.dict(os.environ, {ENV_PREFIX + "BENCH_MODEL": "old_value"}):
            with settings_session(bench_model=None):
                assert os.environ.get(ENV_PREFIX + "BENCH_MODEL") == ""


class TestSettingsValueToEnv:
    """Test _settings_value_to_env conversion."""

    def test_settings_value_to_env_bool_true(self):
        assert _settings_value_to_env(True) == "1"

    def test_settings_value_to_env_bool_false(self):
        assert _settings_value_to_env(False) == "0"

    def test_settings_value_to_env_none(self):
        assert _settings_value_to_env(None) == ""

    def test_settings_value_to_env_string(self):
        assert _settings_value_to_env("some_value") == "some_value"


class TestSyncEnvFromSettings:
    """Test _sync_env_from_settings function."""

    def test_sync_env_from_settings_saves_original_values(self):
        """_sync_env_from_settings should save original env values."""
        os.environ[ENV_COMPUTE_BACKEND] = "original"

        settings = PrivTuneSettings(compute_backend="cuda")
        saved = _sync_env_from_settings(settings)

        assert saved[ENV_COMPUTE_BACKEND] == "original"
        assert os.environ[ENV_COMPUTE_BACKEND] == "cuda"

        _restore_env(saved)
        assert os.environ[ENV_COMPUTE_BACKEND] == "original"

    def test_sync_env_from_settings_handles_unset_vars(self):
        """_sync_env_from_settings should handle unset env vars."""
        os.environ.pop(ENV_COMPUTE_BACKEND, None)

        settings = PrivTuneSettings(compute_backend="rocm")
        saved = _sync_env_from_settings(settings)

        assert saved[ENV_COMPUTE_BACKEND] is None
        assert os.environ[ENV_COMPUTE_BACKEND] == "rocm"

        _restore_env(saved)
        assert os.environ.get(ENV_COMPUTE_BACKEND) is None


class TestDeviceUtilsIntegration:
    """Test interaction with videotuna.utils.device_utils."""

    def test_detect_compute_backend_respects_settings(self):
        """detect_compute_backend should respect VIDEOTUNA_COMPUTE_BACKEND setting."""
        from videotuna.utils.device_utils import detect_compute_backend

        with settings_session(compute_backend="cpu"):
            backend = detect_compute_backend()
            assert backend == "cpu"

    def test_detect_compute_backend_auto_calls_raw(self):
        """detect_compute_backend with 'auto' should use _detect_compute_backend_raw."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch(
            "videotuna.utils.device_utils._detect_compute_backend_raw",
            return_value="cuda",
        ):
            with settings_session(compute_backend="auto"):
                backend = detect_compute_backend()
                assert backend == "cuda"

    def test_detect_compute_backend_via_env_var(self):
        """detect_compute_backend should work via VIDEOTUNA_COMPUTE_BACKEND env var."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch.dict(os.environ, {ENV_COMPUTE_BACKEND: "cpu"}, clear=False):
            backend = detect_compute_backend()
            assert backend == "cpu"

    def test_detect_compute_backend_cpu_explicit(self):
        """detect_compute_backend returns cpu when VIDEOTUNA_COMPUTE_BACKEND=cpu."""
        from videotuna.utils.device_utils import detect_compute_backend

        with settings_session(compute_backend="cpu"):
            assert detect_compute_backend() == "cpu"

    def test_detect_compute_backend_cuda_without_gpu_fails(self):
        """detect_compute_backend with 'cuda' should fail when GPU unavailable."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch(
            "videotuna.utils.device_utils._torch_hip_version", return_value=None
        ):
            with mock.patch(
                "videotuna.utils.device_utils.torch.cuda.is_available",
                return_value=False,
            ):
                with settings_session(compute_backend="cuda"):
                    with pytest.raises(
                        RuntimeError,
                        match="VIDEOTUNA_COMPUTE_BACKEND=cuda but "
                        "torch.cuda.is_available",
                    ):
                        detect_compute_backend()

    def test_detect_compute_backend_rocm_without_hip_fails(self):
        """detect_compute_backend with 'rocm' should fail when PyTorch lacks HIP."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch(
            "videotuna.utils.device_utils._torch_hip_version", return_value=None
        ):
            with settings_session(compute_backend="rocm"):
                with pytest.raises(
                    RuntimeError,
                    match=(
                        "VIDEOTUNA_COMPUTE_BACKEND=rocm but "
                        "PyTorch was not built with HIP"
                    ),
                ):
                    detect_compute_backend()

    def test_detect_compute_backend_rocm_without_gpu_fails(self):
        """detect_compute_backend fails for rocm without a visible GPU."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch(
            "videotuna.utils.device_utils._torch_hip_version", return_value="5.7"
        ):
            with mock.patch(
                "videotuna.utils.device_utils.torch.cuda.is_available",
                return_value=False,
            ):
                with settings_session(compute_backend="rocm"):
                    with pytest.raises(
                        RuntimeError,
                        match=(
                            "VIDEOTUNA_COMPUTE_BACKEND=rocm but "
                            "no ROCm GPU is visible"
                        ),
                    ):
                        detect_compute_backend()

    def test_detect_compute_backend_cuda_with_hip_fails(self):
        """detect_compute_backend with 'cuda' on a HIP build should fail."""
        from videotuna.utils.device_utils import detect_compute_backend

        with mock.patch(
            "videotuna.utils.device_utils._torch_hip_version", return_value="5.7"
        ):
            with mock.patch(
                "videotuna.utils.device_utils.torch.cuda.is_available",
                return_value=True,
            ):
                with settings_session(compute_backend="cuda"):
                    with pytest.raises(
                        RuntimeError,
                        match="VIDEOTUNA_COMPUTE_BACKEND=cuda but PyTorch reports HIP",
                    ):
                        detect_compute_backend()


class TestInferenceSettingsSession:
    """Test inference_settings_session() context manager."""

    def test_cpu_smoke_sets_eager_backend(self):
        from videotuna.settings import inference_settings_session

        with inference_settings_session(cpu_smoke=True) as settings:
            assert settings.attn_backend == "eager"
            assert settings.torch_compile is False
            assert settings.cpu_mode == "smoke"

    def test_inference_settings_session_compile_flag(self):
        from videotuna.settings import inference_settings_session

        with inference_settings_session(compile_flag=True) as settings:
            assert settings.torch_compile is True

    def test_inference_settings_session_neither_flag(self):
        from videotuna.settings import inference_settings_session

        with inference_settings_session() as settings:
            assert settings.torch_compile is False


class TestProjectedEnvKeys:
    """Test that projected (non-VIDEOTUNA) env vars are saved/restored by session."""

    def test_diffsers_attn_backend_restored_on_session_exit(self):
        from videotuna.settings import (
            ENV_DIFFUSERS_ATTN_BACKEND,
            settings_session,
        )

        os.environ.pop(ENV_DIFFUSERS_ATTN_BACKEND, None)

        with settings_session(attn_backend="flash"):
            os.environ[ENV_DIFFUSERS_ATTN_BACKEND] = "flash"

        assert ENV_DIFFUSERS_ATTN_BACKEND not in os.environ

    def test_diffsers_attn_backend_prior_value_restored(self):
        from videotuna.settings import (
            ENV_DIFFUSERS_ATTN_BACKEND,
            settings_session,
        )

        os.environ[ENV_DIFFUSERS_ATTN_BACKEND] = "prior_value"
        try:
            with settings_session(attn_backend="sdpa"):
                os.environ[ENV_DIFFUSERS_ATTN_BACKEND] = "native"

            assert os.environ[ENV_DIFFUSERS_ATTN_BACKEND] == "prior_value"
        finally:
            os.environ.pop(ENV_DIFFUSERS_ATTN_BACKEND, None)

    def test_apply_diffusers_attention_backend_env_fallback(self):
        """apply_diffusers_attention_backend writes DIFFUSERS_ATTN_BACKEND when
        the model lacks set_attention_backend."""
        from videotuna.settings import (
            ENV_DIFFUSERS_ATTN_BACKEND,
            inference_settings_session,
        )
        from videotuna.utils.attention import apply_diffusers_attention_backend

        with inference_settings_session(cpu_smoke=True):
            os.environ.pop(ENV_DIFFUSERS_ATTN_BACKEND, None)
            apply_diffusers_attention_backend(object())
            assert os.environ[ENV_DIFFUSERS_ATTN_BACKEND] == "_native_math"

        assert ENV_DIFFUSERS_ATTN_BACKEND not in os.environ
