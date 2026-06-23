"""Cyclopts CLI for multi-GPU launch validation."""

from __future__ import annotations

import sys

from cyclopts import App

from videotuna.utils.cli_console import install_pretty_tracebacks
from videotuna.utils.multi_gpu_launcher import (
    MultiGpuMode,
    MultiGpuSpec,
    diagnose_failure,
    validate_multi_gpu_setup,
)

app = App(
    name="validate-multi-gpu",
    help="Validate and generate safe multi-GPU launch commands.",
)


@app.command(name="inference")
def validate_inference(
    mode: MultiGpuMode,
    gpu_ids: str,
    config: str | None = None,
    ulysses_degree: int | None = None,
    ring_degree: int | None = None,
    max_memory_per_gpu: str = "22GiB",
    offload_mode: str = "none",
    dry_run: bool = False,
    **extra_args: str,
) -> None:
    """Validate multi-GPU inference setup and optionally print a safe launch command.

    Parameters
    ----------
    mode
        multi-GPU path: device_map (Diffusers) or xfuser (native sequence parallel)
    gpu_ids
        Comma-separated GPU indices to use (e.g. "0,1")
    config
        Path to inference YAML config
    ulysses_degree
        Ulysses attention degree (xfuser mode only)
    ring_degree
        Ring attention degree (xfuser mode only)
    max_memory_per_gpu
        Per-GPU memory budget for device_map mode (e.g. "22GiB")
    offload_mode
        CPU offload mode: none, model, sequential (device_map requires none)
    dry_run
        Validate and print command without executing
    extra_args
        Additional CLI flags to include in the generated command
    """
    _do_validate(
        mode=mode,
        gpu_ids=_parse_gpu_ids(gpu_ids),
        config_path=config,
        ulysses_degree=ulysses_degree or 1,
        ring_degree=ring_degree or 1,
        max_memory_per_gpu=max_memory_per_gpu,
        offload_mode=offload_mode,
        dry_run=dry_run,
        extra_args=dict(extra_args),
    )


@app.command(name="training")
def validate_training(
    mode: MultiGpuMode,
    gpu_ids: str,
    config: str | None = None,
    devices: str = "0,",
    num_processes: int = 1,
    dry_run: bool = False,
    **extra_args: str,
) -> None:
    """Validate multi-GPU training setup and optionally print a safe launch command.

    Parameters
    ----------
    mode
        Training mode: wan_lightning or flux_accelerate
    gpu_ids
        Comma-separated GPU indices to use (e.g. "0,1,2,3")
    config
        Path to training YAML/JSON config
    devices
        Lightning trainer devices string (wan_lightning mode, e.g. "0,1,2,3")
    num_processes
        Number of accelerate processes (flux_accelerate mode)
    dry_run
        Validate and print command without executing
    extra_args
        Additional CLI flags to include in the generated command
    """
    _do_validate(
        mode=mode,
        gpu_ids=_parse_gpu_ids(gpu_ids),
        config_path=config,
        devices=devices,
        num_processes=num_processes,
        dry_run=dry_run,
        extra_args=dict(extra_args),
    )


@app.command(name="diagnose")
def diagnose(symptom: str) -> None:
    """Look up troubleshooting steps for a known failure symptom.

    Parameters
    ----------
    symptom
        Short description (e.g. "hang", "oom", "xfuser_import_error", "xfuser_rocm")
    """
    steps = diagnose_failure(symptom)
    print(f"Troubleshooting: {symptom}")
    print(50 * "-")
    for step in steps:
        print(f"  - {step}")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_gpu_ids(raw: str) -> tuple[int, ...]:
    """Parse comma-separated GPU IDs into a tuple of ints."""
    try:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        print(f"Error: invalid --gpu-ids {raw!r}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _do_validate(
    *,
    mode: str,
    gpu_ids: tuple[int, ...],
    config_path: str | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    max_memory_per_gpu: str = "22GiB",
    offload_mode: str = "none",
    devices: str = "0,",
    num_processes: int = 1,
    dry_run: bool = False,
    extra_args: dict[str, str] | None = None,
) -> None:
    spec = MultiGpuSpec(
        mode=mode,  # type: ignore[arg-type]
        gpu_ids=gpu_ids,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        config_path=config_path,
        max_memory_per_gpu=max_memory_per_gpu,
        offload_mode=offload_mode,
        devices=devices,
        num_processes=num_processes,
        extra_args=extra_args or {},
    )

    result = validate_multi_gpu_setup(spec)
    _print_results(result, dry_run=dry_run)

    if not result.success:
        raise SystemExit(2)


def _print_results(
    result,
    *,
    dry_run: bool = False,
) -> None:
    import sys as _sys

    for diag in result.diagnostics:
        prefix = {
            "fatal": "ERROR",
            "warning": "WARN",
            "info": "INFO",
        }.get(diag.severity, "INFO")
        out = _sys.stderr if diag.severity == "fatal" else _sys.stdout
        print(f"[{prefix}] {diag.message}", file=out)

    if result.generated_command and dry_run:
        print(50 * "-")
        print("Generated launch command:")
        print()
        print(f"  {result.generated_command}")
        print()
        print("Copy and paste the command above to launch multi-GPU inference.")
    elif result.generated_command:
        print()
        print("No errors. Run with the default command or add --dry-run to preview.")


install_pretty_tracebacks()


def main() -> None:
    """Entry point for validate-multi-gpu CLI."""
    raise SystemExit(app(sys.argv[1:]))


if __name__ == "__main__":
    main()
