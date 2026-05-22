"""Shared path conventions for the data layer."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (assumes this file lives in src/artifact_limits/data)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    p = data_root() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = data_root() / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def splits_dir() -> Path:
    p = data_root() / "splits"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = repo_root() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    p = results_dir() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir() -> Path:
    p = results_dir() / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p
