"""Shared subprocess launcher for AlphaTrade JAX GPU jobs."""

from __future__ import annotations

import json
import os
import sys
from typing import Mapping, Sequence


DEFAULT_XLA_FLAGS = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="


def gpu_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment configured for JAX CUDA subprocesses."""
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env.update(
        {
            "JAX_PLATFORMS": "cuda",
            "XLA_FLAGS": DEFAULT_XLA_FLAGS,
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-alphatrade")
    if extra:
        env.update(extra)
    return env


def build_module_cmd(
    module: str,
    cli_args: Sequence[str],
    *,
    gpu: bool,
    package: str = "alphatrade.scripts",
) -> tuple[list[str], dict[str, str] | None]:
    """Build a subprocess command for an AlphaTrade script module.

    GPU mode uses the historical import-based launch path to keep JAX CUDA
    plugin initialization before script-level imports.
    """
    if gpu:
        argv_str = json.dumps(["run", *map(str, cli_args)])
        code = (
            f"import sys; sys.argv = {argv_str}; "
            f"from {package}.{module} import main; main()"
        )
        return [sys.executable, "-c", code], gpu_env()

    script = f"src/alphatrade/scripts/{module}.py"
    return [sys.executable, script, *map(str, cli_args)], None
