"""Focused tests for the Phase-1 acceptance experiment."""

from __future__ import annotations

from pathlib import Path

from experiments import phase1_acceptance


def test_fixed_seed_round_trip_render_and_replay(tmp_path: Path) -> None:
    """
    The acceptance helpers persist and render a deterministic track and replay it.
    """
    track = phase1_acceptance.generate_save_reload_render(0, tmp_path)

    assert track.generation.seed == 0
    phase1_acceptance.assert_deterministic_replay(0)


def test_manual_driver_starts_and_quits_with_dummy_sdl() -> None:
    """
    The interactive driver can initialize and quit without a physical display.
    """
    phase1_acceptance.manual_driver_smoke(0)
