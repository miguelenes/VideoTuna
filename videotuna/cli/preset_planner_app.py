"""Standalone CLI for the hardware-aware preset planner."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Annotated, Any

from cyclopts import App, Parameter

from videotuna.utils.cli_console import install_pretty_tracebacks
from videotuna.utils.preset_planner import (
    FlowName,
    PresetPlanningError,
    plan_preset,
    preflight_check,
)

app = App(
    name="plan-preset",
    help="Recommend or validate a PrivTune inference preset for your hardware.",
)


@Parameter(name="*")
@dataclass
class PlannerOptions:
    """Options for the preset planner CLI."""

    flow: FlowName = "wan_t2v"
    preset: Annotated[str | None, Parameter(name="preset")] = None
    cpu_smoke: Annotated[bool, Parameter(name="cpu-smoke")] = False
    compile: bool = False
    transformer_quant: str | None = None
    vram_gb: Annotated[float | None, Parameter(name="vram-gb")] = None
    json: bool = False


def _format_recommendation(rec: Any) -> str:
    lines = [
        f"Recommended preset: {rec.preset_path}",
        f"  Flow: {rec.flow}",
        f"  Tier: {rec.tier.value}",
        f"  Memory preset: {rec.memory_preset}",
        f"  Dtype: {rec.dtype}",
        f"  Offload: {rec.offload_mode}",
        f"  Attention: {rec.attn_backend}",
    ]
    if rec.transformer_quant not in (None, "none"):
        lines.append(f"  Quant: {rec.transformer_quant} ({rec.quant_backend})")
    lines.append(f"  Compile: {rec.compile_enabled}")
    if rec.vram_gb is not None:
        lines.append(f"  Detected VRAM: {rec.vram_gb:.1f} GB")
    if rec.warnings:
        lines.append("  Warnings:")
        lines.extend(f"    - {w}" for w in rec.warnings)
    if rec.hints:
        lines.append("  Hints:")
        lines.extend(f"    - {h}" for h in rec.hints)
    return "\n".join(lines)


def _recommend(options: PlannerOptions) -> int:
    """Recommend a preset for the current hardware."""
    try:
        rec = plan_preset(
            options.flow,
            transformer_quant=options.transformer_quant,
            compile_flag=options.compile,
            cpu_smoke=options.cpu_smoke,
            vram_gb=options.vram_gb,
        )
    except PresetPlanningError as exc:
        if options.json:
            payload = {
                "ok": False,
                "error": exc.message,
                "hints": exc.hints,
                "detected_backend": exc.detected_backend,
                "detected_vram_gb": exc.detected_vram_gb,
            }
            print(json.dumps(payload, indent=2))
        else:
            print(exc.format(), file=sys.stderr)
        return 2

    if options.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(_format_recommendation(rec))
    return 0


def _validate(options: PlannerOptions) -> int:
    """Validate a chosen preset YAML against the current hardware."""
    if not options.preset:
        print("Error: --preset <path> is required for validate.", file=sys.stderr)
        return 2

    try:
        rec = preflight_check(
            options.preset,
            compile_flag=options.compile,
            cpu_smoke=options.cpu_smoke,
            vram_gb=options.vram_gb,
        )
    except PresetPlanningError as exc:
        if options.json:
            payload = {
                "ok": False,
                "error": exc.message,
                "hints": exc.hints,
                "detected_backend": exc.detected_backend,
                "detected_vram_gb": exc.detected_vram_gb,
            }
            print(json.dumps(payload, indent=2))
        else:
            print(exc.format(), file=sys.stderr)
        return 2

    if options.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print("Preset is compatible with the current hardware.")
        print(_format_recommendation(rec))
    return 0


@app.default
def _default(options: PlannerOptions) -> int:
    """Default command: recommend unless --preset is given, then validate."""
    if options.preset:
        return _validate(options)
    return _recommend(options)


@app.command(name="recommend")
def recommend(options: PlannerOptions) -> int:
    """Recommend a preset for the current hardware."""
    return _recommend(options)


@app.command(name="validate")
def validate(options: PlannerOptions) -> int:
    """Validate a chosen preset YAML against the current hardware."""
    return _validate(options)


@app.command(name="list-flows")
def list_flows() -> int:
    """List supported inference flows and their available preset tiers."""
    from videotuna.utils.preset_planner import _PRESET_REGISTRY

    data: dict[str, list[str]] = {
        flow: [tier.value for tier in tiers] for flow, tiers in _PRESET_REGISTRY.items()
    }
    print(json.dumps(data, indent=2))
    return 0


def main() -> int:
    """Entry point for the plan-preset Poetry script."""
    install_pretty_tracebacks()
    return app(sys.argv[1:])


install_pretty_tracebacks()

if __name__ == "__main__":
    raise SystemExit(main())
