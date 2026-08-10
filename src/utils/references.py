"""Reference policies and deterministic environment evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from envs.racing import RacingEnv

from .metrics import (
    EpisodeOutcome,
    EpisodeRecord,
    MetricScope,
    RunCategory,
    ScalarSummary,
    TransitionRecord,
)
from .seeding import RunSeedStreams, SeedStream


class ReferencePolicy(Protocol):
    """
    Select a normalized racing action from one observation.
    """

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return a normalized throttle/brake and steering action.
        """
        ...


@dataclass(frozen=True, slots=True)
class ScriptedFrenetController:
    """
    Apply the documented deterministic Frenet driving controller.

    Fields:
        * lateral_gain: Steering correction for lateral distance.
        * heading_gain: Steering correction for heading error.
        * curvature_gain: Steering feed-forward from preview curvature.
        * lateral_acceleration_limit: Curvature-dependent speed target constant.
        * maximum_target_speed: Upper bound for the target speed.
        * speed_error_scale: Speed error that saturates throttle/braking.
    """

    lateral_gain: float = 0.15
    heading_gain: float = 0.8
    curvature_gain: float = 50.0
    lateral_acceleration_limit: float = 20.0
    maximum_target_speed: float = 50.0
    speed_error_scale: float = 10.0

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return the exact documented deterministic controller action.
        """
        lateral_distance, heading_error, speed, curvature = map(float, observation)
        steering = np.clip(
            -self.lateral_gain * lateral_distance
            - self.heading_gain * heading_error
            + self.curvature_gain * curvature,
            -1.0,
            1.0,
        )
        target_speed = min(
            self.maximum_target_speed,
            np.sqrt(self.lateral_acceleration_limit / max(abs(curvature), 1e-4)),
        )
        throttle = np.clip((target_speed - speed) / self.speed_error_scale, -1.0, 1.0)
        return np.asarray((throttle, steering), dtype=np.float32)


@dataclass(slots=True)
class RandomActionReference:
    """
    Sample normalized actions from only the named evaluation/reference stream.

    Fields:
        * generator: Isolated NumPy generator used for reference actions.
    """

    generator: np.random.Generator

    def action(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Return one independent uniform action without consuming global randomness.
        """
        del observation
        return self.generator.uniform(-1.0, 1.0, size=2).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ReferenceEvaluation:
    """
    Store an evaluation summary and its action-level semantic trajectory.

    Fields:
        * episode: Complete semantic episode summary.
        * transitions: Transition records in action order.
    """

    episode: EpisodeRecord
    transitions: tuple[TransitionRecord, ...]


def random_action_reference(streams: RunSeedStreams) -> RandomActionReference:
    """
    Build a random policy from the run's isolated evaluation/reference stream.
    """
    return RandomActionReference(
        streams.numpy_generator(SeedStream.EVALUATION_REFERENCE)
    )


def evaluate_reference(
    environment: RacingEnv,
    policy: ReferencePolicy,
    *,
    run_category: RunCategory,
    episode_index: int = 0,
    training_interactions: int = 0,
    evaluation_interactions_before: int = 0,
    reset_seed: int | None = None,
    circuit_identity: str | None = None,
    root_identity: int | None = None,
    circuit_split: str | None = None,
) -> ReferenceEvaluation:
    """
    Run one isolated reference episode and retain every semantic transition.
    """
    observation, _ = environment.reset(seed=reset_seed)
    total_return = 0.0
    maximum_progress = 0.0
    transitions: list[TransitionRecord] = []
    speeds: list[float] = []
    throttles: list[float] = []
    absolute_steering: list[float] = []
    resolved_circuit_identity = circuit_identity or str(
        environment.track.generation.seed
    )
    for step_index in range(environment.config.simulation.max_episode_steps):
        action = policy.action(observation)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        speeds.append(float(observation[2]))
        throttles.append(float(action[0]))
        absolute_steering.append(abs(float(action[1])))
        progress = float(info["episode_progress"]) / environment.track.track_length
        maximum_progress = max(maximum_progress, progress)
        transitions.append(
            TransitionRecord(
                run_category=run_category,
                scope=MetricScope.REFERENCE,
                episode_index=episode_index,
                step_index=step_index,
                observation=tuple(float(value) for value in observation),
                next_observation=tuple(float(value) for value in next_observation),
                action=(float(action[0]), float(action[1])),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                collision=bool(info["collision"]),
                lap_completed=bool(info["lap_completed"]),
                progress=progress,
                elapsed_time=float(info["elapsed_time"]),
                circuit_identity=resolved_circuit_identity,
            )
        )
        total_return += float(reward)
        observation = next_observation
        if terminated or truncated:
            outcome = _episode_outcome(terminated, truncated, info)
            interactions = len(transitions)
            episode = EpisodeRecord(
                run_category=run_category,
                scope=MetricScope.REFERENCE,
                episode_index=episode_index,
                outcome=outcome,
                undiscounted_return=total_return,
                training_target_total=None,
                interactions=interactions,
                simulated_time=float(info["elapsed_time"]),
                final_progress=progress,
                maximum_progress=maximum_progress,
                lap_time=(
                    float(info["elapsed_time"])
                    if outcome is EpisodeOutcome.COMPLETED
                    else None
                ),
                training_interactions=training_interactions,
                evaluation_interactions=evaluation_interactions_before + interactions,
                circuit_identity=resolved_circuit_identity,
                root_identity=root_identity,
                observation_type="frenet",
                circuit_seed=environment.track.generation.seed,
                circuit_split=circuit_split,
                speed=_scalar_summary(speeds),
                throttle=_scalar_summary(throttles),
                absolute_steering=_scalar_summary(absolute_steering),
                positive_throttle_fraction=float(np.mean(np.asarray(throttles) > 0)),
                braking_fraction=float(np.mean(np.asarray(throttles) < 0)),
            )
            return ReferenceEvaluation(episode=episode, transitions=tuple(transitions))
    raise RuntimeError("RacingEnv did not end within its configured episode limit.")


def _scalar_summary(values: list[float]) -> ScalarSummary:
    """
    Summarize one non-empty episode signal using population dispersion.
    """
    array = np.asarray(values, dtype=np.float64)
    return ScalarSummary(
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
    )


def _episode_outcome(
    terminated: bool,
    truncated: bool,
    info: dict[str, object],
) -> EpisodeOutcome:
    """
    Convert the environment's terminal flags into one explicit outcome.
    """
    if terminated and bool(info["lap_completed"]):
        return EpisodeOutcome.COMPLETED
    if terminated and bool(info["collision"]):
        return EpisodeOutcome.CRASHED
    if truncated:
        return EpisodeOutcome.TIME_LIMIT
    raise ValueError("Terminal transition lacks a supported RacingEnv outcome.")
