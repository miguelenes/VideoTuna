"""Tests for cyclopts inference CLI groups and Poetry entrypoints."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from videotuna.cli.inference_app import (
    PRESET_DOMAIN_T2I,
    PRESET_VALIDATE_I2V,
    PRESET_VALIDATE_T2V,
    PRESET_WAN2_2_I2V_720P,
    PRESET_WAN2_2_T2V_720P,
    _entry_for_preset,
    _make_app,
    _run_inference_with_options,
    app,
    inference_domain_t2i_entry,
    inference_flux_lora_entry,
    inference_wan2_2_i2v_720p_entry,
    inference_wan2_2_t2v_720p_entry,
    main,
    validate_domain_i2v_entry,
    validate_domain_t2v_entry,
)
from videotuna.cli.inference_options import (
    InferenceRunConfig,
    InferenceRunOptions,
    StandardInferenceOptions,
    inference_options_to_config,
    validate_preset_requirements,
)


def _help_text(command: list[str]) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (f"from videotuna.cli.inference_app import app; app({command!r})"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
        ["inference-wan2.2-t2v-720p", "--help"],
    ],
)
def test_inference_help_lists_shared_flags(command: list[str]) -> None:
    help_text = _help_text(command)
    for flag in (
        "--lorackpt",
        "--memory-preset",
        "--enable_vae_tiling",
        "--num-inference-steps",
    ):
        assert flag in help_text


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
    ],
)
def test_inference_help_exposes_canonical_quant_and_omits_legacy_fp8(
    command: list[str],
) -> None:
    help_text = _help_text(command)
    for flag in ("--transformer-quant", "--quant-backend"):
        assert flag in help_text
    for forbidden in ("enable_fp8", "enable-fp8", "dit-weight", "hunyuan"):
        assert forbidden not in help_text.lower()


def test_flag_parity_across_presets() -> None:
    run = InferenceRunOptions(lorackpt="/tmp/lora", num_inference_steps=8)
    standard = StandardInferenceOptions(memory_preset="balanced", device="cuda:0")
    t2i = inference_options_to_config(
        run=run,
        standard=standard,
        preset=PRESET_DOMAIN_T2I,
    ).model_dump()
    wan = inference_options_to_config(
        run=run,
        standard=standard,
        preset=PRESET_WAN2_2_T2V_720P,
    ).model_dump()

    shared_keys = {k for k in t2i if k not in {"config", "enable_model_cpu_offload"}}
    for key in shared_keys:
        assert t2i[key] == wan[key], key

    assert t2i["config"] == PRESET_DOMAIN_T2I.config
    assert wan["config"] == PRESET_WAN2_2_T2V_720P.config
    assert t2i["enable_model_cpu_offload"] is True
    assert wan["enable_model_cpu_offload"] is False


def test_domain_t2i_preset_applies_without_user_config() -> None:
    run_config = inference_options_to_config(preset=PRESET_DOMAIN_T2I)
    assert run_config.config == PRESET_DOMAIN_T2I.config
    assert run_config.enable_model_cpu_offload is True


def test_validate_domain_t2v_requires_checkpoint() -> None:
    with pytest.raises(SystemExit) as exc:
        validate_preset_requirements(InferenceRunOptions(), PRESET_VALIDATE_T2V)
    assert exc.value.code == 2


def test_cyclopts_parses_standard_flags() -> None:
    captured: dict[str, object] = {}

    def handler(
        run: InferenceRunOptions | None = None,
        *,
        standard: StandardInferenceOptions | None = None,
    ) -> None:
        captured["run_config"] = inference_options_to_config(run=run, standard=standard)

    probe = app.__class__(name="probe")
    probe.command(name="probe")(handler)
    probe(
        [
            "probe",
            "--lorackpt",
            "/tmp/lora",
            "--memory-preset",
            "low_vram",
            "--enable_vae_tiling",
            "--config",
            "configs/inference/presets/flux_domain_lora_smoke.yaml",
        ]
    )
    run_config = captured["run_config"]
    assert isinstance(run_config, InferenceRunConfig)
    assert run_config.lorackpt == "/tmp/lora"
    assert run_config.memory_preset == "low_vram"
    assert run_config.enable_vae_tiling is True


def test_inference_run_config_round_trip() -> None:
    run_config = inference_options_to_config(preset=PRESET_DOMAIN_T2I)
    round_tripped = InferenceRunConfig.model_validate(
        run_config.model_dump(mode="json")
    )
    assert round_tripped.config == run_config.config
    assert round_tripped.enable_model_cpu_offload == run_config.enable_model_cpu_offload


# ---------------------------------------------------------------------------
# Inference app preset wiring and entrypoint registration (CPU-only)
# ---------------------------------------------------------------------------


class TestInferenceAppPresets:
    """Verify preset wiring and entrypoint registration without model execution."""

    def test_all_presets_have_required_fields(self) -> None:
        presets = [
            PRESET_DOMAIN_T2I,
            PRESET_VALIDATE_T2V,
            PRESET_WAN2_2_T2V_720P,
            PRESET_VALIDATE_I2V,
            PRESET_WAN2_2_I2V_720P,
        ]
        for preset in presets:
            assert isinstance(preset.cli_name, str)
            assert isinstance(preset.config, str)
            assert isinstance(preset.enable_model_cpu_offload, bool)
            assert isinstance(preset.require_checkpoint, bool)
            assert isinstance(preset.require_prompt_dir, bool)

    def test_preset_configs_exist_on_disk(self) -> None:
        import os

        presets = [
            PRESET_DOMAIN_T2I,
            PRESET_VALIDATE_T2V,
            PRESET_WAN2_2_T2V_720P,
            PRESET_VALIDATE_I2V,
            PRESET_WAN2_2_I2V_720P,
        ]
        for preset in presets:
            assert os.path.isfile(
                preset.config
            ), f"Preset config {preset.config} not found on disk"

    def test_domain_t2i_preset_has_cpu_offload(self) -> None:
        assert PRESET_DOMAIN_T2I.enable_model_cpu_offload is True

    def test_validate_t2v_requires_checkpoint(self) -> None:
        assert PRESET_VALIDATE_T2V.require_checkpoint is True
        assert PRESET_VALIDATE_T2V.enable_model_cpu_offload is True

    def test_wan22_720p_preset_no_cpu_offload(self) -> None:
        assert PRESET_WAN2_2_T2V_720P.enable_model_cpu_offload is False
        assert PRESET_WAN2_2_T2V_720P.require_checkpoint is False

    def test_validate_i2v_requires_checkpoint_and_prompt_dir(self) -> None:
        assert PRESET_VALIDATE_I2V.require_checkpoint is True
        assert PRESET_VALIDATE_I2V.require_prompt_dir is True
        assert PRESET_VALIDATE_I2V.enable_model_cpu_offload is True


def test_make_app_creates_app_with_preset_name() -> None:
    app_instance = _make_app(PRESET_DOMAIN_T2I)
    assert app_instance.name[0] == "inference-domain-t2i"
    assert app_instance.default_command is not None


def test_make_app_registers_default_command() -> None:
    app_instance = _make_app(PRESET_VALIDATE_T2V)
    assert app_instance.name[0] == "validate-domain-t2v"


def test_entry_for_preset_returns_callable() -> None:
    entry = _entry_for_preset(PRESET_DOMAIN_T2I)
    assert callable(entry)
    assert entry.__name__ == "inference_domain_t2i"


def test_entry_for_preset_validate_t2v_naming() -> None:
    entry = _entry_for_preset(PRESET_VALIDATE_T2V)
    assert entry.__name__ == "validate_domain_t2v"


def test_entry_for_preset_wan22_720p_naming() -> None:
    entry = _entry_for_preset(PRESET_WAN2_2_T2V_720P)
    assert entry.__name__ == "inference_wan2_2_t2v_720p"


def test_inference_flux_lora_is_same_as_domain_t2i() -> None:
    assert inference_flux_lora_entry is inference_domain_t2i_entry


def test_shared_app_has_expected_commands() -> None:
    names = set(app._commands.keys())
    expected = {
        "inference-domain-t2i",
        "validate-domain-t2v",
        "inference-wan2.2-t2v-720p",
        "run",
    }
    assert expected.issubset(names)


class TestInferenceAppRunMocked:
    """CPU-only wiring tests for _run_inference_with_options with mocked execution."""

    def test_run_inference_with_options_calls_run_inference(self) -> None:
        with mock.patch("scripts.inference_new.run_inference") as mock_run:
            run = InferenceRunOptions(lorackpt="/tmp/lora")
            standard = StandardInferenceOptions(memory_preset="low_vram")
            _run_inference_with_options(run, standard, preset=PRESET_DOMAIN_T2I)
            mock_run.assert_called_once()
            config = mock_run.call_args[0][0]
            assert isinstance(config, InferenceRunConfig)
            assert config.config == PRESET_DOMAIN_T2I.config
            assert config.lorackpt == "/tmp/lora"
            assert config.memory_preset == "low_vram"

    def test_run_inference_with_options_without_preset(self) -> None:
        with mock.patch("scripts.inference_new.run_inference") as mock_run:
            run = InferenceRunOptions(
                config="configs/inference/presets/flux_domain_lora_smoke.yaml"
            )
            _run_inference_with_options(run, None)
            mock_run.assert_called_once()
            config = mock_run.call_args[0][0]
            assert (
                config.config == "configs/inference/presets/flux_domain_lora_smoke.yaml"
            )

    def test_run_inference_with_options_validates_preset(self) -> None:
        with mock.patch("scripts.inference_new.run_inference") as mock_run:
            with pytest.raises(SystemExit) as exc:
                _run_inference_with_options(
                    InferenceRunOptions(), None, preset=PRESET_VALIDATE_T2V
                )
            assert exc.value.code == 2
            mock_run.assert_not_called()

    def test_preset_command_runs_via_app_mocked(self) -> None:
        with mock.patch("scripts.inference_new.run_inference") as mock_run:
            app(["inference-domain-t2i", "--lorackpt", "/tmp/lora"])
            mock_run.assert_called_once()
            config = mock_run.call_args[0][0]
            assert config.config == PRESET_DOMAIN_T2I.config


def test_main_function_exists() -> None:
    assert callable(main)
    assert main.__name__ == "main"


def test_all_public_entries_are_callable() -> None:
    entries = [
        inference_domain_t2i_entry,
        validate_domain_t2v_entry,
        inference_wan2_2_t2v_720p_entry,
        validate_domain_i2v_entry,
        inference_wan2_2_i2v_720p_entry,
    ]
    for entry in entries:
        assert callable(entry)


def test_preset_wan22_i2v_720p_fields() -> None:
    assert PRESET_WAN2_2_I2V_720P.cli_name == "inference-wan2.2-i2v-720p"
    assert PRESET_WAN2_2_I2V_720P.enable_model_cpu_offload is False
    assert PRESET_WAN2_2_I2V_720P.require_checkpoint is False


def test_make_app_without_preset_uses_custom_name() -> None:
    custom = _make_app(name="my-app")
    assert custom.name[0] == "my-app"


def test_make_app_without_preset_fallback_name() -> None:
    fallback = _make_app()
    assert fallback.name[0] == "privtune-inference"


def test_entry_for_preset_all_presets() -> None:
    presets = [
        PRESET_DOMAIN_T2I,
        PRESET_VALIDATE_T2V,
        PRESET_WAN2_2_T2V_720P,
        PRESET_VALIDATE_I2V,
        PRESET_WAN2_2_I2V_720P,
    ]
    for preset in presets:
        entry = _entry_for_preset(preset)
        assert callable(entry)
        assert entry.__name__ == preset.cli_name.replace(".", "_").replace("-", "_")


def test_make_app_default_command_is_not_none() -> None:
    app_with_default = _make_app(PRESET_DOMAIN_T2I)
    assert app_with_default.default_command is not None


def test_make_app_custom_name_overrides_preset_cli_name() -> None:
    custom = _make_app(PRESET_DOMAIN_T2I, name="custom-name")
    assert custom.name[0] == "custom-name"


def test_inference_run_options_default_preset_config() -> None:
    run_config = inference_options_to_config(preset=PRESET_DOMAIN_T2I)
    assert run_config.config == "configs/inference/presets/flux_domain_lora_smoke.yaml"
    assert run_config.enable_model_cpu_offload is True


def test_inference_run_options_preset_validate_t2v_config() -> None:
    run_config = inference_options_to_config(preset=PRESET_VALIDATE_T2V)
    assert (
        run_config.config == "configs/inference/presets/wan_domain_lora_smoke_22.yaml"
    )
    assert run_config.enable_model_cpu_offload is True


def test_inference_run_options_preset_validate_i2v_config() -> None:
    run_config = inference_options_to_config(preset=PRESET_VALIDATE_I2V)
    assert run_config.config == "configs/inference/presets/wan_domain_i2v_smoke_22.yaml"
    assert run_config.enable_model_cpu_offload is True


def test_inference_run_options_no_config_raises() -> None:
    with pytest.raises(ValueError, match="requires a YAML config"):
        inference_options_to_config()


# ---------------------------------------------------------------------------
# End-to-end typed inference path — no argparse.Namespace bridge
# ---------------------------------------------------------------------------


_REMOVED_BRIDGE = [
    ("videotuna.cli.inference_options", "inference_options_to_namespace"),
    ("videotuna.utils.args_utils", "prepare_inference_args"),
    ("videotuna.utils.inference_cli", "prepare_cli_inference_args"),
    ("videotuna.utils.inference_cli", "apply_compile_env"),
    ("videotuna.utils.inference_cli", "apply_cpu_smoke_env"),
]


@pytest.mark.parametrize(("module_name", "attr"), _REMOVED_BRIDGE)
def test_removed_bridge_functions_are_gone(module_name: str, attr: str) -> None:
    """Guard against reintroducing the deprecated Namespace bridge."""
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, attr), f"{module_name}.{attr} should have been removed"


def test_inference_new_has_no_namespace() -> None:
    """scripts/inference_new.py must not import argparse or use Namespace."""
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "inference_new.py"
    ).read_text(encoding="utf-8")
    assert "import argparse" not in source
    assert "argparse.Namespace" not in source


def test_run_inference_consumes_run_config() -> None:
    """run_inference's first parameter must be annotated InferenceRunConfig."""
    from scripts.inference_new import run_inference

    sig = inspect.signature(run_inference)
    first_param = next(iter(sig.parameters.values()))
    assert first_param.annotation is InferenceRunConfig


def test_prepare_inference_config_propagates_preset_side_effects() -> None:
    """Regression: preset side effects (low_vram → dtype=fp16, sequential offload)
    must propagate into the merged YAML inference_config, not be lost."""
    from unittest import mock

    from omegaconf import OmegaConf

    from videotuna.utils.args_utils import prepare_inference_config

    run_config = InferenceRunConfig(
        config="configs/inference/presets/flux_domain_lora_smoke.yaml",
        memory_preset="low_vram",
        savedir="/tmp/videotuna-test-preset",
    )
    yaml_config = OmegaConf.load(run_config.config)

    with (
        mock.patch("videotuna.utils.device_utils.gpu_is_available", return_value=True),
        mock.patch(
            "videotuna.utils.device_utils.detect_compute_backend", return_value="cuda"
        ),
        mock.patch("videotuna.utils.device_utils.resolve_cpu_mode", return_value="off"),
        mock.patch(
            "videotuna.utils.args_utils.process_savedir", side_effect=lambda d: d
        ),
        mock.patch(
            "videotuna.utils.cli_console.render_inference_config_panel"
        ) as mock_panel,
    ):
        merged = prepare_inference_config(run_config, yaml_config)

    inference = merged.inference
    assert inference.dtype == "fp16"
    assert inference.enable_sequential_cpu_offload is True
    assert inference.enable_model_cpu_offload is False
    mock_panel.assert_not_called()


def test_typed_end_to_end_merge() -> None:
    """inference_options_to_config → prepare_inference_config produces a DictConfig
    whose inference section reflects CLI overrides."""
    from unittest import mock

    from omegaconf import OmegaConf

    from videotuna.utils.args_utils import prepare_inference_config

    run_config = inference_options_to_config(
        run=InferenceRunOptions(
            lorackpt="/tmp/lora",
            num_inference_steps=4,
            seed=123,
        ),
        preset=PRESET_DOMAIN_T2I,
    )
    yaml_config = OmegaConf.load(run_config.config)

    with (
        mock.patch("videotuna.utils.device_utils.gpu_is_available", return_value=True),
        mock.patch(
            "videotuna.utils.device_utils.detect_compute_backend", return_value="cuda"
        ),
        mock.patch("videotuna.utils.device_utils.resolve_cpu_mode", return_value="off"),
        mock.patch(
            "videotuna.utils.args_utils.process_savedir", side_effect=lambda d: d
        ),
        mock.patch(
            "videotuna.utils.cli_console.render_inference_config_panel"
        ) as mock_panel,
    ):
        merged = prepare_inference_config(run_config, yaml_config)

    inference = merged.inference
    assert inference.lorackpt == "/tmp/lora"
    assert int(inference.num_inference_steps) == 4
    assert int(inference.seed) == 123
    mock_panel.assert_not_called()
