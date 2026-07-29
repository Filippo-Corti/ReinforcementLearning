"""Smoke tests for the initial project scaffold."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ("configs", "envs", "gymnasium", "numpy", "pygame"),
)
def test_scaffold_module_is_importable(module_name: str) -> None:
    """Every package required by the Phase-1 scaffold can be imported."""
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
