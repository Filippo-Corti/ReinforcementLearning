"""Episode evaluation for scripted and random baseline policies.

Not part of the trained-agent evaluation path: baseline policies are a
reference point computed outside training, only for experiments and tests, so
this lives beside the other shared utilities rather than inside the
`evaluation` package that trained runs depend on.
"""

from __future__ import annotations

import numpy as np

from agents.models import Policy
from envs.racing import RacingEnv
from envs.tracks import track_geometry_summary
from evaluation import trajectory_state
from recording.records import (
    EpisodeOutcome,
    EpisodeRecord,
    LoggedTransitionRecord,
    MetricScope,
    PolicyEvaluationRecord,
    RunCategory,
)

from .statistics import scalar_summary


def evaluate_policy_episode(
    environment: RacingEnv,
    policy: Policy,
    *,
    run_category: RunCategory,
    episode_index: int = 0,
    training_interactions: int = 0,
    evaluation_interactions_before: int = 0,
    reset_seed: int | None = None,
    circuit_identity: str | None = None,
    root_identity: int | None = None,
    circuit_split: str | None = None,
) -> PolicyEvaluationRecord:
    """
    Run one baseline-policy episode and retain its summary and raw trajectory.
    """
    observation, _ = environment.reset(seed=reset_seed)
    total_return = 0.0
    maximum_progress = 0.0
    transitions: list[LoggedTransitionRecord] = []
    speeds: list[float] = []
    throttles: list[float] = []
    absolute_steering: list[float] = []
    resolved_circuit_identity = circuit_identity or str(
        environment.track.generation.seed
    )
    for step_index in range(environment.config.simulation.max_episode_steps):
        current_state = trajectory_state(environment, observation)
        action = policy.action(observation)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        speeds.append(float(observation[2]))
        throttles.append(float(action[0]))
        absolute_steering.append(abs(float(action[1])))
        progress = float(info["episode_progress"]) / environment.track.track_length
        maximum_progress = max(maximum_progress, progress)
        transitions.append(
            LoggedTransitionRecord(
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
                position=current_state.position,
                heading=current_state.heading,
                current_curvature=current_state.current_curvature,
                preview_curvature=current_state.preview_curvature,
                speed=current_state.speed,
                lateral_acceleration_proxy=(current_state.lateral_acceleration_proxy),
            )
        )
        total_return += float(reward)
        observation = next_observation
        if terminated or truncated:
            outcome = EpisodeOutcome.from_transition(
                terminated=terminated, truncated=truncated, info=info
            )
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
                speed=scalar_summary(speeds),
                throttle=scalar_summary(throttles),
                absolute_steering=scalar_summary(absolute_steering),
                positive_throttle_fraction=float(np.mean(np.asarray(throttles) > 0)),
                braking_fraction=float(np.mean(np.asarray(throttles) < 0)),
                circuit_geometry=track_geometry_summary(environment.track),
            )
            return PolicyEvaluationRecord(
                episode=episode, transitions=tuple(transitions)
            )
    raise RuntimeError("RacingEnv did not end within its configured episode limit.")
