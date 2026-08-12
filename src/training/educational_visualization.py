"""Live rollout dashboards and aligned plots for the educational notebooks."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import ArrayLike, NDArray

from configs import EnvironmentConfig
from envs.racing import RacingEnv
from envs.tracks import TrackWithGeometry

from .normalization import RunningObservationNormalizer


class DeterministicPolicy(Protocol):
    """
    Describe the deterministic action operation needed by the rollout viewer.
    """

    def deterministic_action(
        self, normalized_observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Return one bounded deterministic action.
        """
        ...


@dataclass(slots=True)
class _RolloutDashboard:
    """
    Hold the mutable artists updated by one live rollout viewer.

    Fields:
        * figure: Figure containing the environment and statistic axes.
        * display_handle: Notebook display handle, or none for window mode.
        * image: Environment image artist used by inline mode.
        * metric_text: Current progress, return, speed, and control readout.
        * speed_axis: Axis containing the speed history.
        * speed_line: Line containing the speed history.
        * controls_axis: Axis containing throttle and steering histories.
        * throttle_line: Line containing the throttle history.
        * steering_line: Line containing the steering history.
    """

    figure: Figure
    display_handle: Any
    image: AxesImage | None
    metric_text: Text
    speed_axis: Axes
    speed_line: Line2D
    controls_axis: Axes
    throttle_line: Line2D
    steering_line: Line2D


def watch_deterministic_rollout(
    title: str,
    track: TrackWithGeometry,
    agent: DeterministicPolicy,
    normalizer: RunningObservationNormalizer,
    environment_config: EnvironmentConfig,
    *,
    max_steps: int,
    reset_seed: int,
    viewer_mode: str = "inline",
    frame_delay: float = 0.01,
) -> dict[str, float | int | bool]:
    """
    Render a deterministic rollout beside live progress, speed, and controls.

    Inline mode places the environment and statistics in one updating notebook
    figure. Window mode keeps the Pygame environment window and a synchronized
    Matplotlib statistics window.
    """
    if max_steps <= 0:
        raise ValueError("The rollout viewer requires a positive step count.")
    if frame_delay < 0.0:
        raise ValueError("Frame delay cannot be negative.")
    if viewer_mode not in {"inline", "window"}:
        raise ValueError("Viewer mode must be 'inline' or 'window'.")

    render_mode = "rgb_array" if viewer_mode == "inline" else "human"
    environment = RacingEnv(
        track,
        config=environment_config,
        render_mode=render_mode,
    )
    observation, info = environment.reset(seed=reset_seed)
    total_return = 0.0
    speeds = [float(observation[2])]
    throttles = [0.0]
    steering = [0.0]
    step = 0
    figure, dashboard = _rollout_dashboard(
        title,
        environment,
        environment_config,
        viewer_mode,
    )

    try:
        for step in range(1, max_steps + 1):
            normalized_observation = normalizer.normalize(observation)
            action = agent.deterministic_action(normalized_observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += float(reward)
            speeds.append(float(observation[2]))
            throttles.append(float(action[0]))
            steering.append(float(action[1]))

            if viewer_mode == "window":
                environment.render()
            _update_rollout_dashboard(
                dashboard,
                title=title,
                step=step,
                max_steps=max_steps,
                frame=(environment.render() if viewer_mode == "inline" else None),
                progress=float(info["episode_progress"]) / track.track.track_length,
                total_return=total_return,
                speeds=speeds,
                throttles=throttles,
                steering=steering,
            )
            if frame_delay:
                time.sleep(frame_delay)
            if terminated or truncated:
                break
    finally:
        environment.close()
        plt.close(figure)

    return {
        "interactions": step,
        "return": total_return,
        "collision": bool(info["collision"]),
        "lap_completed": bool(info["lap_completed"]),
        "progress": float(info["episode_progress"]) / track.track.track_length,
        "final_speed": speeds[-1],
        "final_throttle": throttles[-1],
        "final_steering": steering[-1],
    }


def trailing_average(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """
    Return the trailing mean using the available prefix before the window fills.
    """
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(
        [
            np.mean(array[max(0, index - window + 1) : index + 1])
            for index in range(array.size)
        ],
        dtype=np.float64,
    )


def plot_task_performance(
    episode_rows: Sequence[Mapping[str, object]],
    evaluation_rows: Sequence[Mapping[str, float | int]],
    *,
    moving_average_window: int,
) -> Figure:
    """
    Plot training returns and greedy return means with one-standard-deviation bands.
    """
    episodes = _episode_indices(episode_rows)
    returns = _episode_values(episode_rows, "undiscounted_return")
    evaluation_interactions = _evaluation_values(
        evaluation_rows, "training_interactions"
    )
    evaluation_returns = _evaluation_values(evaluation_rows, "return_mean")
    evaluation_return_deviations = _evaluation_values(
        evaluation_rows, "return_standard_deviation"
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(episodes, returns, alpha=0.35, label="training episode")
    axes[0].plot(
        episodes,
        trailing_average(returns, moving_average_window),
        linewidth=2.2,
        label=f"moving average (w={moving_average_window})",
    )
    axes[0].set(title="Training return", xlabel="episode", ylabel="return")
    axes[0].legend()

    axes[1].plot(
        evaluation_interactions,
        evaluation_returns,
        marker="o",
        linewidth=2,
        label="greedy mean",
    )
    axes[1].fill_between(
        evaluation_interactions,
        evaluation_returns - evaluation_return_deviations,
        evaluation_returns + evaluation_return_deviations,
        alpha=0.2,
        label=r"mean $\pm$ one standard deviation",
    )
    axes[1].set(
        title="Greedy evaluation return",
        xlabel="training interactions",
        ylabel="return",
    )
    axes[1].legend()
    _style_axes(axes)
    return figure


def plot_progress_and_efficiency(
    episode_rows: Sequence[Mapping[str, object]],
    evaluation_rows: Sequence[Mapping[str, float | int]],
    *,
    moving_average_window: int,
) -> Figure:
    """
    Plot training and greedy progress plus training episode length.
    """
    episodes = _episode_indices(episode_rows)
    progress = _episode_values(episode_rows, "maximum_progress")
    lengths = _episode_values(episode_rows, "interactions")
    evaluation_interactions = _evaluation_values(
        evaluation_rows, "training_interactions"
    )
    evaluation_progress = _evaluation_values(evaluation_rows, "maximum_progress_mean")
    evaluation_progress_deviations = _evaluation_values(
        evaluation_rows, "maximum_progress_standard_deviation"
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    axes[0].plot(episodes, progress, alpha=0.35, label="training episode")
    axes[0].plot(
        episodes,
        trailing_average(progress, moving_average_window),
        linewidth=2.2,
        label=f"moving average (w={moving_average_window})",
    )
    axes[0].set(
        title="Training maximum progress",
        xlabel="episode",
        ylabel="lap fraction",
    )
    axes[0].legend()

    axes[1].plot(
        evaluation_interactions,
        evaluation_progress,
        marker="o",
        alpha=0.45,
        label="greedy mean",
    )
    axes[1].plot(
        evaluation_interactions,
        trailing_average(evaluation_progress, moving_average_window),
        linewidth=2.2,
        label=f"moving average (w={moving_average_window})",
    )
    axes[1].fill_between(
        evaluation_interactions,
        evaluation_progress - evaluation_progress_deviations,
        evaluation_progress + evaluation_progress_deviations,
        alpha=0.16,
        label=r"mean $\pm$ one standard deviation",
    )
    axes[1].set(
        title="Greedy maximum progress",
        xlabel="training interactions",
        ylabel="lap fraction",
    )
    axes[1].legend()

    axes[2].plot(episodes, lengths, alpha=0.35, label="training episode")
    axes[2].plot(
        episodes,
        trailing_average(lengths, moving_average_window),
        linewidth=2.2,
        label=f"moving average (w={moving_average_window})",
    )
    axes[2].set(
        title="Training episode length",
        xlabel="episode",
        ylabel="interactions",
    )
    axes[2].legend()
    _style_axes(axes)
    return figure


def plot_driving_behavior(
    episode_rows: Sequence[Mapping[str, object]],
    *,
    moving_average_window: int,
) -> Figure:
    """
    Plot mean episode speed and both the signed and absolute throttle action.

    The magnitude alone cannot distinguish a learned throttle from exploration
    noise: a zero mean squashed by `tanh` already produces a large and perfectly
    flat magnitude. The signed mean is the curve that moves only when the policy
    itself has learned to accelerate.
    """
    episodes = _episode_indices(episode_rows)
    speeds = _episode_values(episode_rows, "mean_speed")
    signed_throttles = _episode_values(episode_rows, "mean_throttle")
    throttle_magnitudes = _episode_values(episode_rows, "mean_throttle_magnitude")
    figure, axes = plt.subplots(1, 3, figsize=(19, 4.5), constrained_layout=True)
    for axis, values, title, ylabel in (
        (axes[0], speeds, "Mean training speed", "speed"),
        (
            axes[1],
            signed_throttles,
            "Mean signed throttle",
            "throttle/brake",
        ),
        (
            axes[2],
            throttle_magnitudes,
            "Mean throttle magnitude",
            "absolute throttle/brake",
        ),
    ):
        axis.plot(episodes, values, alpha=0.35, label="training episode")
        axis.plot(
            episodes,
            trailing_average(values, moving_average_window),
            linewidth=2.2,
            label=f"moving average (w={moving_average_window})",
        )
        axis.set(title=title, xlabel="episode", ylabel=ylabel)
        axis.legend()
    axes[1].axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    _style_axes(axes)
    return figure


def plot_outcomes_and_lap_time(
    episode_rows: Sequence[Mapping[str, object]],
    *,
    moving_average_window: int,
) -> Figure:
    """
    Plot the training outcome mix and the lap times of completed episodes.

    A return curve cannot say whether a policy is crashing, idling or lapping
    slowly, and a lap that completes says nothing about how fast it was. These
    are the two questions the reward alone leaves ambiguous.
    """
    episodes = _episode_indices(episode_rows)
    outcomes = [str(row["outcome"]) for row in episode_rows]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    for name in ("completed", "crashed", "stalled", "time_limit"):
        indicator = np.asarray(
            [1.0 if outcome == name else 0.0 for outcome in outcomes],
            dtype=np.float64,
        )
        if not indicator.any():
            continue
        axes[0].plot(
            episodes,
            trailing_average(indicator, moving_average_window),
            linewidth=2.0,
            label=name,
        )
    axes[0].set(
        title="Training outcome mix",
        xlabel="episode",
        ylabel=f"fraction (w={moving_average_window})",
        ylim=(-0.02, 1.02),
    )
    axes[0].legend()

    completed = [
        (index, _as_float(row["lap_time"]))
        for index, row in zip(episodes, episode_rows, strict=True)
        if row.get("lap_time") is not None
    ]
    if completed:
        completed_episodes, lap_times = zip(*completed, strict=True)
        axes[1].plot(
            completed_episodes, lap_times, alpha=0.35, label="completed episode"
        )
        axes[1].plot(
            completed_episodes,
            trailing_average(np.asarray(lap_times), moving_average_window),
            linewidth=2.2,
            label=f"moving average (w={moving_average_window})",
        )
    axes[1].set(title="Completed lap time", xlabel="episode", ylabel="seconds")
    if completed:
        axes[1].legend()
    _style_axes(axes)
    return figure


def plot_optimization_and_exploration(
    update_rows: Sequence[Mapping[str, object]],
    *,
    algorithm_name: str,
    moving_average_window: int,
    include_critic: bool,
) -> Figure:
    """
    Plot losses, parameter norms, and learned throttle/steering exploration scales.
    """
    interactions = np.asarray(
        [_as_float(row["training_interactions"]) for row in update_rows],
        dtype=np.float64,
    )
    actor_losses = _diagnostic_values(update_rows, "actor_loss")
    actor_weight_norms = _diagnostic_values(update_rows, "actor_weight_norm")
    sigma_throttle = np.exp(_diagnostic_values(update_rows, "log_standard_deviation_0"))
    sigma_steering = np.exp(_diagnostic_values(update_rows, "log_standard_deviation_1"))

    columns = 4 if include_critic else 3
    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(5.2 * columns, 4.5),
        constrained_layout=True,
    )
    axes[0].plot(interactions, actor_losses, alpha=0.4, label="per update")
    axes[0].plot(
        interactions,
        trailing_average(actor_losses, moving_average_window),
        linewidth=2.2,
        label=f"moving average (w={moving_average_window})",
    )
    axes[0].set(
        title=f"{algorithm_name} actor loss",
        xlabel="training interactions",
        ylabel="loss",
    )
    axes[0].legend()

    norm_axis_index = 1
    if include_critic:
        critic_losses = _diagnostic_values(update_rows, "critic_loss")
        axes[1].plot(interactions, critic_losses, alpha=0.4, label="per update")
        axes[1].plot(
            interactions,
            trailing_average(critic_losses, moving_average_window),
            linewidth=2.2,
            label=f"moving average (w={moving_average_window})",
        )
        axes[1].set(
            title=f"{algorithm_name} critic loss",
            xlabel="training interactions",
            ylabel="loss",
        )
        axes[1].legend()
        norm_axis_index = 2

    axes[norm_axis_index].plot(
        interactions, actor_weight_norms, linewidth=2, label="actor"
    )
    if include_critic:
        axes[norm_axis_index].plot(
            interactions,
            _diagnostic_values(update_rows, "critic_weight_norm"),
            linewidth=2,
            label="critic",
        )
    axes[norm_axis_index].set(
        title="Parameter weight norm",
        xlabel="training interactions",
        ylabel="Euclidean norm",
    )
    axes[norm_axis_index].legend()

    sigma_axis = axes[-1]
    sigma_axis.plot(
        interactions, sigma_throttle, linewidth=2, label=r"$\sigma_{throttle}$"
    )
    sigma_axis.plot(
        interactions, sigma_steering, linewidth=2, label=r"$\sigma_{steering}$"
    )
    sigma_axis.set(
        title="Learned Gaussian exploration scale",
        xlabel="training interactions",
        ylabel=r"$\sigma = \exp(\ell)$",
    )
    sigma_axis.legend()
    _style_axes(axes)
    return figure


def plot_algorithm_diagnostics(
    update_rows: Sequence[Mapping[str, object]],
    *,
    algorithm_name: str,
) -> Figure | None:
    """
    Plot diagnostics that are meaningful only for A2C or PPO.
    """
    interactions = np.asarray(
        [_as_float(row["training_interactions"]) for row in update_rows],
        dtype=np.float64,
    )
    if algorithm_name == "A2C":
        panels = (
            ("advantage_standard_deviation", "Raw advantage standard deviation"),
            ("explained_variance", "Critic explained variance"),
        )
    elif algorithm_name == "PPO":
        panels = (
            ("approximate_kl", "Approximate KL"),
            ("clip_fraction", "Clipped sample fraction"),
            ("explained_variance", "Critic explained variance"),
        )
    else:
        return None

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 4.2),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, (key, title) in zip(axes_array, panels, strict=True):
        axis.plot(interactions, _diagnostic_values(update_rows, key), linewidth=1.8)
        axis.set(title=title, xlabel="training interactions", ylabel="value")
    _style_axes(axes_array)
    return figure


def _rollout_dashboard(
    title: str,
    environment: RacingEnv,
    environment_config: EnvironmentConfig,
    viewer_mode: str,
) -> tuple[Figure, _RolloutDashboard]:
    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.45, 1.0))
    track_axis = figure.add_subplot(grid[:, 0]) if viewer_mode == "inline" else None
    metrics_axis = figure.add_subplot(grid[0, 1])
    speed_axis = figure.add_subplot(grid[1, 1])
    controls_axis = figure.add_subplot(grid[2, 1])

    image = None
    if track_axis is not None:
        frame = environment.render()
        if frame is None:
            raise RuntimeError("Inline rendering requires an RGB frame.")
        image = track_axis.imshow(frame)
        track_axis.set_title(title)
        track_axis.axis("off")
    metrics_axis.axis("off")
    metric_text = metrics_axis.text(0.0, 0.95, "", va="top", family="monospace")
    speed_line = speed_axis.plot([], [], linewidth=2, color="tab:blue")[0]
    speed_axis.set(
        title="Speed over time",
        xlabel="interaction",
        ylabel="speed",
        ylim=(0.0, environment_config.vehicle.max_speed),
    )
    throttle_line = controls_axis.plot(
        [], [], linewidth=2, label="throttle / brake", color="tab:green"
    )[0]
    steering_line = controls_axis.plot(
        [], [], linewidth=2, label="steering", color="tab:orange"
    )[0]
    controls_axis.set(
        title="Controls over time",
        xlabel="interaction",
        ylabel="normalized action",
        ylim=(-1.05, 1.05),
    )
    controls_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    controls_axis.legend(loc="upper right")
    _style_axes((speed_axis, controls_axis))

    display_handle = (
        display(figure, display_id=True) if viewer_mode == "inline" else None
    )
    if viewer_mode == "window":
        plt.show(block=False)
    return figure, _RolloutDashboard(
        figure=figure,
        display_handle=display_handle,
        image=image,
        metric_text=metric_text,
        speed_axis=speed_axis,
        speed_line=speed_line,
        controls_axis=controls_axis,
        throttle_line=throttle_line,
        steering_line=steering_line,
    )


def _update_rollout_dashboard(
    dashboard: _RolloutDashboard,
    *,
    title: str,
    step: int,
    max_steps: int,
    frame: NDArray[np.uint8] | None,
    progress: float,
    total_return: float,
    speeds: Sequence[float],
    throttles: Sequence[float],
    steering: Sequence[float],
) -> None:
    if frame is not None and dashboard.image is not None:
        dashboard.image.set_data(frame)
    dashboard.metric_text.set_text(
        "\n".join(
            (
                title,
                f"step       {step:4d} / {max_steps}",
                f"progress   {progress:8.3%}",
                f"return     {total_return:8.2f}",
                f"speed      {speeds[-1]:8.3f}",
                f"throttle   {throttles[-1]:8.3f}",
                f"steering   {steering[-1]:8.3f}",
            )
        )
    )
    x = np.arange(len(speeds))
    dashboard.speed_line.set_data(x, speeds)
    dashboard.throttle_line.set_data(x, throttles)
    dashboard.steering_line.set_data(x, steering)
    dashboard.speed_axis.set_xlim(0, max(1, len(speeds) - 1))
    dashboard.controls_axis.set_xlim(0, max(1, len(speeds) - 1))

    if dashboard.display_handle is not None:
        dashboard.display_handle.update(dashboard.figure)
    else:
        dashboard.figure.canvas.draw_idle()
        dashboard.figure.canvas.flush_events()


def _episode_indices(
    rows: Sequence[Mapping[str, object]],
) -> NDArray[np.float64]:
    return np.asarray([_as_float(row["episode_index"]) for row in rows])


def _episode_values(
    rows: Sequence[Mapping[str, object]], key: str
) -> NDArray[np.float64]:
    return np.asarray([_as_float(row[key]) for row in rows], dtype=np.float64)


def _evaluation_values(
    rows: Sequence[Mapping[str, float | int]], key: str
) -> NDArray[np.float64]:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _diagnostic_values(
    rows: Sequence[Mapping[str, object]], key: str
) -> NDArray[np.float64]:
    values = []
    for row in rows:
        diagnostics = row["diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise TypeError("Update diagnostics must be a mapping.")
        values.append(_as_float(diagnostics[key]))
    return np.asarray(values, dtype=np.float64)


def _as_float(value: object) -> float:
    return float(cast(float | int, value))


def _style_axes(axes: Sequence[object] | NDArray[np.object_]) -> None:
    for axis in np.asarray(axes, dtype=object).flat:
        axis.grid(alpha=0.25)
