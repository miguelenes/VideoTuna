"""Tests for Diffusers pipeline quantization helpers."""

from unittest import mock

import pytest

from videotuna.utils.diffusers_quantization import (
    build_pipeline_quantization_config,
    normalize_quant_backend,
    normalize_transformer_quant,
    reject_enable_fp8_for_non_hunyuan,
    resolve_quant_components,
    validate_transformer_quant,
)


def test_resolve_quant_components_wan_22():
    components = resolve_quant_components("wan", "2.2", "t2v")
    assert components == ["transformer", "transformer_2"]


def test_resolve_quant_components_flux():
    assert resolve_quant_components("flux", None, "t2i") == ["transformer"]


def test_normalize_transformer_quant_defaults():
    assert normalize_transformer_quant(None) == "none"
    assert normalize_transformer_quant("int8_wo") == "int8_wo"


def test_normalize_transformer_quant_invalid():
    with pytest.raises(ValueError, match="Unsupported transformer_quant"):
        normalize_transformer_quant("uint7")


def test_normalize_quant_backend_defaults():
    assert normalize_quant_backend(None) == "torchao"
    assert normalize_quant_backend("quanto") == "quanto"


def test_build_pipeline_quantization_config_none():
    assert (
        build_pipeline_quantization_config(
            transformer_quant="none",
            quant_backend="torchao",
            components=["transformer"],
        )
        is None
    )


def test_build_pipeline_quantization_config_wan_torchao():
    mock_torchao_cfg = mock.MagicMock()
    mock_pipe_cfg = mock.MagicMock()
    with mock.patch(
        "videotuna.utils.diffusers_quantization._build_torchao_component_config",
        return_value=mock_torchao_cfg,
    ):
        with mock.patch(
            "diffusers.PipelineQuantizationConfig",
            return_value=mock_pipe_cfg,
        ) as pipe_cfg_cls:
            cfg = build_pipeline_quantization_config(
                transformer_quant="int8_wo",
                quant_backend="torchao",
                components=["transformer", "transformer_2"],
            )
    assert cfg is mock_pipe_cfg
    pipe_cfg_cls.assert_called_once()
    mapping = pipe_cfg_cls.call_args.kwargs["quant_mapping"]
    assert set(mapping.keys()) == {"transformer", "transformer_2"}
    assert mapping["transformer"] is mock_torchao_cfg


def test_validate_transformer_quant_rejects_cpu():
    with mock.patch(
        "videotuna.utils.diffusers_quantization.detect_compute_backend",
        return_value="cpu",
    ):
        with pytest.raises(RuntimeError, match="not supported on CPU"):
            validate_transformer_quant(
                transformer_quant="int8_wo",
                quant_backend="torchao",
                offload_mode="sequential",
            )


def test_validate_transformer_quant_rejects_rocm():
    with mock.patch(
        "videotuna.utils.diffusers_quantization.detect_compute_backend",
        return_value="rocm",
    ):
        with pytest.raises(RuntimeError, match="not supported on AMD ROCm"):
            validate_transformer_quant(
                transformer_quant="int8_wo",
                quant_backend="torchao",
                offload_mode="model",
            )


def test_validate_transformer_quant_fp8_requires_ada():
    with mock.patch(
        "videotuna.utils.diffusers_quantization.detect_compute_backend",
        return_value="cuda",
    ):
        with mock.patch(
            "videotuna.utils.diffusers_quantization.gpu_is_available",
            return_value=True,
        ):
            with mock.patch(
                "videotuna.utils.diffusers_quantization.torch.cuda.get_device_capability",
                return_value=(8, 0),
            ):
                with pytest.raises(RuntimeError, match="fp8_wo requires"):
                    validate_transformer_quant(
                        transformer_quant="fp8_wo",
                        quant_backend="torchao",
                        offload_mode="none",
                    )


def test_validate_transformer_quant_quanto_import_guard():
    with mock.patch(
        "videotuna.utils.diffusers_quantization.detect_compute_backend",
        return_value="cuda",
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "optimum.quanto" or name.startswith("optimum"):
                raise ImportError("no quanto")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="optimum-quanto"):
                validate_transformer_quant(
                    transformer_quant="int8_wo",
                    quant_backend="quanto",
                    offload_mode="model",
                )


def test_reject_enable_fp8_for_diffusers_flow():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="transformer-quant fp8_wo"):
        reject_enable_fp8_for_non_hunyuan(
            "videotuna.flow.diffusers_video.DiffusersVideoFlow",
            SimpleNamespace(),
        )


def test_maybe_adjust_offload_for_quant():
    from argparse import Namespace

    from videotuna.utils.diffusers_quantization import maybe_adjust_offload_for_quant

    args = Namespace(
        enable_sequential_cpu_offload=True,
        enable_model_cpu_offload=False,
    )
    maybe_adjust_offload_for_quant(args, "int8_wo")
    assert args.enable_sequential_cpu_offload is False
    assert args.enable_model_cpu_offload is True
