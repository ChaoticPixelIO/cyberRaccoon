"""Locate the project root directory at runtime.

Used by modules that need to access project-level resources (skills/, scripts/)
that live outside the installed package. Finds the project root by walking up
from the current file until it finds pyproject.toml.

Usage::

    from cyberraccoon._project_root import PROJECT_ROOT

    skills_dir = PROJECT_ROOT / "skills"
    scripts_dir = PROJECT_ROOT / "scripts"
"""
from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Walk upward from this file to find the directory containing pyproject.toml.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If pyproject.toml is not found in any ancestor directory.
    """
    current = Path(__file__).resolve().parent
    for ancestor in [current] + list(current.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    raise RuntimeError(
        "Could not find project root (no pyproject.toml in any ancestor of "
        f"{Path(__file__).resolve()})"
    )


PROJECT_ROOT = find_project_root()
