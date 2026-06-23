"""Rich-based CLI output helpers (panels, tracebacks) complementing cyclopts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich import traceback as rich_traceback
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from omegaconf import DictConfig

_console = Console(stderr=True)
_tracebacks_installed = False


def install_pretty_tracebacks() -> None:
    """Enable Rich-formatted tracebacks for CLI entrypoints."""
    global _tracebacks_installed
    if _tracebacks_installed:
        return
    rich_traceback.install(show_locals=False)
    _tracebacks_installed = True


def _inference_config_items(inference_config: DictConfig) -> dict[str, Any]:
    return {
        "Mode": inference_config.mode,
        "Save Directory": inference_config.savedir,
        "Height": inference_config.height,
        "Width": inference_config.width,
        "Frames": inference_config.frames,
        "FPS": inference_config.fps,
        "Seed": inference_config.seed,
        "Sample Batch Size": inference_config.bs,
        "Samples per Prompt": inference_config.n_samples_prompt,
    }


def render_inference_config_panel(
    inference_config: DictConfig,
    *,
    console: Console | None = None,
) -> None:
    """Render a bordered inference-config summary panel to the CLI console."""
    out = console or _console
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Value")

    for key, value in _inference_config_items(inference_config).items():
        if value is not None:
            table.add_row(key, str(value))

    out.print(
        Panel(
            table,
            title="Inference Configuration",
            border_style="cyan",
            expand=False,
        )
    )
