"""Visualize how the racing reward scales with progress and lap time."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from configs import RewardConfig, SimulationConfig

# The coefficients this project started from, kept so the plots can show what
# changed rather than only what the reward is now. Standing still cost less than
# any attempt to drive, and one lap of shaping was worth a twentieth of a crash.
# Length of the seed-0 circuit, used only to turn a lap fraction into a
# plausible number of elapsed agent steps.
REFERENCE_TRACK_LENGTH = 318.0

ORIGINAL_REWARD = RewardConfig(
    finish_reward=10.0,
    lap_time_bonus=0.0,
    crash_penalty=20.0,
    time_penalty_rate=0.05,
    progress_coefficient=1.0,
)


@dataclass(frozen=True, slots=True)
class RewardModel:
    """
    Evaluate undiscounted episode returns for whole behaviours, not single steps.

    A per-step reward says little on its own: what decides a policy's fate is
    the return of the behaviour it induces. Every method below therefore scores
    a complete episode.

    Fields:
        * reward: The reward coefficients under test.
        * simulation: Episode clock and timestep settings.
    """

    reward: RewardConfig
    simulation: SimulationConfig

    @property
    def step_cost(self) -> float:
        """
        Return what one agent step costs in time penalty.
        """
        return self.reward.time_penalty_rate * self.simulation.agent_timestep

    def idling(self) -> float:
        """
        Return the score of never leaving the start line.
        """
        return -self.step_cost * self.simulation.max_episode_steps

    def crash_after(self, lap_fraction: float, steps: int) -> float:
        """
        Return the score of reaching a lap fraction and then leaving the track.
        """
        return (
            self.reward.progress_coefficient * lap_fraction
            - self.step_cost * steps
            - self.reward.crash_penalty
        )

    def completed_lap(self, steps: int) -> float:
        """
        Return the score of one completed lap taking a given number of steps.
        """
        unused_clock = 1.0 - steps / self.simulation.max_episode_steps
        return (
            self.reward.progress_coefficient
            - self.step_cost * steps
            + self.reward.finish_reward
            + self.reward.lap_time_bonus * unused_clock
        )


def _plot_progress(axis, models: dict[str, RewardModel], speed: float) -> None:
    """
    Plot the return of crashing after a given lap fraction against that fraction.
    """
    fractions = np.linspace(0.0, 1.0, 200)
    for name, model in models.items():
        steps = fractions * _lap_steps(model, speed)
        axis.plot(
            fractions,
            [
                model.crash_after(float(f), round(float(s)))
                for f, s in zip(fractions, steps, strict=True)
            ],
            linewidth=2.0,
            label=f"{name}: crash after progress",
        )
        axis.axhline(
            model.idling(),
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{name}: never move",
        )
    axis.set(
        title="Getting further before crashing",
        xlabel="lap fraction reached",
        ylabel="episode return",
    )
    axis.legend(fontsize=8)


def _plot_lap_time(axis, models: dict[str, RewardModel]) -> None:
    """
    Plot the return of a completed lap against the time it took.
    """
    for name, model in models.items():
        steps = np.arange(100, model.simulation.max_episode_steps, 5)
        seconds = steps * model.simulation.agent_timestep
        returns = [model.completed_lap(int(s)) for s in steps]
        axis.plot(seconds, returns, linewidth=2.0, label=f"{name}: completed lap")
        axis.axhline(
            -model.reward.crash_penalty,
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label=f"{name}: crash at once",
        )
    axis.set(
        title="Completing the lap faster",
        xlabel="lap time (s)",
        ylabel="episode return",
    )
    axis.legend(fontsize=8)


def _plot_ordering(axis, models: dict[str, RewardModel], speed: float) -> None:
    """
    Plot the four behaviours whose ordering decides whether the task is learnable.
    """
    labels = (
        "never move",
        "crash at once",
        "crash at half a lap",
        "slow lap (28 s)",
        "fast lap (9 s)",
    )
    width = 0.38
    positions = np.arange(len(labels))
    for offset, (name, model) in zip((-width / 2, width / 2), models.items()):
        lap_steps = _lap_steps(model, speed)
        scores = (
            model.idling(),
            model.crash_after(0.0, 1),
            model.crash_after(0.5, int(lap_steps * 0.5)),
            model.completed_lap(700),
            model.completed_lap(232),
        )
        bars = axis.bar(positions + offset, scores, width, label=name)
        axis.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set(title="Ordering of whole behaviours", ylabel="episode return")
    axis.set_xticks(positions, labels, rotation=20, ha="right", fontsize=8)
    axis.legend(fontsize=8)


def _lap_steps(model: RewardModel, speed: float) -> float:
    """
    Return how many agent steps a lap takes at one average speed.
    """
    return REFERENCE_TRACK_LENGTH / speed / model.simulation.agent_timestep


def build_figure(speed: float) -> Figure:
    """
    Build the three-panel comparison of the current and original coefficients.
    """
    simulation = SimulationConfig()
    models = {
        "current": RewardModel(RewardConfig(), simulation),
        "original": RewardModel(ORIGINAL_REWARD, replace(simulation)),
    }
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    _plot_progress(axes[0], models, speed)
    _plot_lap_time(axes[1], models)
    _plot_ordering(axes[2], models, speed)
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Racing reward: what each whole behaviour is worth", fontsize=13)
    return figure


def main() -> None:
    """
    Render the reward-scaling figure to a file or an interactive window.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speed",
        type=float,
        default=20.0,
        help="average speed used to convert lap fraction into elapsed steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the figure here instead of opening a window.",
    )
    parsed = parser.parse_args()

    figure = build_figure(parsed.speed)
    if parsed.output is None:
        plt.show()
        return
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(parsed.output, dpi=140)
    print(f"Wrote {parsed.output}")


if __name__ == "__main__":
    main()
