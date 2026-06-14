"""Runtime output path helpers for AlphaTrade experiments.

The AlphaTrade code lives inside this repository, but generated experiment
artifacts should live outside it by default.
"""

from __future__ import annotations

import os
from pathlib import Path


_ENV_RUNS_ROOT = "ALPHATRADE_RUNS_ROOT"


def repo_root() -> Path:
    """Return the repository root containing this package."""
    return Path(__file__).resolve().parents[2]


def default_runs_root() -> Path:
    """Return the default external runs root.

    Users can override this with ALPHATRADE_RUNS_ROOT. Without an override, we
    place generated outputs in a sibling directory next to this repository.
    """
    env_value = os.environ.get(_ENV_RUNS_ROOT)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root().parent / "alphatrade_runs" / "default").resolve()


def resolve_output_root(output_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an output root from a CLI value or the default runs root."""
    if output_root:
        return Path(output_root).expanduser().resolve()
    return default_runs_root()


def add_output_args(parser) -> None:
    """Add standard generated-output path arguments to a CLI parser."""
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Root for generated outputs "
            "(default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)"
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Reports directory (default: <output-root>/reports)",
    )


def resolve_child_dir(
    *,
    output_root: str | os.PathLike[str] | None,
    explicit_dir: str | os.PathLike[str] | None,
    child_name: str,
) -> Path:
    """Resolve a generated-output child directory."""
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()
    return resolve_output_root(output_root) / child_name


def reports_dir(
    output_root: str | os.PathLike[str] | None = None,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_child_dir(
        output_root=output_root, explicit_dir=explicit_dir, child_name="reports"
    )


def checkpoints_dir(
    output_root: str | os.PathLike[str] | None = None,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_child_dir(
        output_root=output_root,
        explicit_dir=explicit_dir,
        child_name="checkpoints",
    )


def artifacts_dir(
    output_root: str | os.PathLike[str] | None = None,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_child_dir(
        output_root=output_root, explicit_dir=explicit_dir, child_name="artifacts"
    )


def cache_dir(
    output_root: str | os.PathLike[str] | None = None,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_child_dir(
        output_root=output_root, explicit_dir=explicit_dir, child_name="cache"
    )
