"""Train one project-owned on-policy learner on a saved racing circuit."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from agents import A2CAgent, ParameterizedOnPolicyAgent, PPOAgent, ReinforceAgent
from configs import (
    FIXED_CRITIC_CONFIG,
    LARGE_ACTOR_CONFIG,
    MEDIUM_ACTOR_CONFIG,
    SMALL_ACTOR_CONFIG,
    A2CConfig,
    ActorConfig,
    Algorithm,
    CriticConfig,
    EnvironmentConfig,
    ExecutionConfig,
    ObservationRepresentation,
    PPOConfig,
    ReinforceConfig,
    StartStateConfig,
    TrainingConfig,
)
from envs.racing import RacingEnv, observation_space_for
from envs.tracks import TrackWithGeometry
from recording import (
    MetricScope,
    ResourceRecord,
    RunCategory,
    RunDirectory,
    TimingRecord,
    UpdateRecord,
    collect_run_metadata,
)
from training import (
    A2CTrainingEngine,
    CircuitSplit,
    EvaluationCircuit,
    PPOTrainingEngine,
    ReinforceTrainingEngine,
    RunningObservationNormalizer,
    TrainingCircuitSchedule,
    TrainingEngine,
    circuit_track_seed,
)
from utils.random import (
    RunSeedStreams,
    SeedNamespace,
    SeedStream,
    configure_torch_determinism,
)

_ACTORS: dict[str, ActorConfig] = {
    "small": SMALL_ACTOR_CONFIG,
    "medium": MEDIUM_ACTOR_CONFIG,
    "large": LARGE_ACTOR_CONFIG,
}

# Each algorithm owns its collection strategy, so each owns its engine. They
# share a constructor signature, and everything that is not collection.
_ENGINES: dict[Algorithm, type[TrainingEngine]] = {
    Algorithm.REINFORCE: ReinforceTrainingEngine,
    Algorithm.A2C: A2CTrainingEngine,
    Algorithm.PPO: PPOTrainingEngine,
}


def run_reinforce_training(
    *,
    seed: int,
    track_path: str | Path | None = None,
    run_path: str | Path,
    actor_config: ActorConfig,
    actor_learning_rate: float,
    training_interaction_budget: int,
    environment_config: EnvironmentConfig | None = None,
    reinforce_config: ReinforceConfig | None = None,
    evaluation_interval: int | None = None,
    near_saturated_steering_threshold: float | None = None,
    execution_config: ExecutionConfig | None = None,
    training_circuit_schedule: TrainingCircuitSchedule | None = None,
    evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    final_evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    training_reference_circuits: int = 0,
    observation: ObservationRepresentation = ObservationRepresentation.FRENET,
    run_category: RunCategory = RunCategory.REDUCED_VALIDATION,
) -> TrainingEngine:
    """
    Train REINFORCE through the shared engine and persist its run records.
    """
    if training_interaction_budget <= 0:
        raise ValueError("Training interaction budget must be positive.")
    if actor_learning_rate <= 0:
        raise ValueError("Actor learning rate must be positive.")
    environment_config = environment_config or EnvironmentConfig()
    reinforce_config = reinforce_config or ReinforceConfig()
    execution_config = execution_config or ExecutionConfig()
    configure_torch_determinism(execution_config)
    resolved_actor_config = replace(actor_config, learning_rate=actor_learning_rate)
    base_training_config = TrainingConfig(actor=resolved_actor_config)
    resolved_evaluation_interval = (
        base_training_config.evaluation.evaluation_interval
        if evaluation_interval is None
        else evaluation_interval
    )
    training_config = replace(
        base_training_config,
        reinforce=reinforce_config,
        training_interaction_budget=training_interaction_budget,
        evaluation=replace(
            base_training_config.evaluation,
            evaluation_interval=resolved_evaluation_interval,
        ),
        logging=replace(
            base_training_config.logging,
            near_saturated_steering_threshold=(
                base_training_config.logging.near_saturated_steering_threshold
                if near_saturated_steering_threshold is None
                else near_saturated_steering_threshold
            ),
        ),
        execution=execution_config,
    )
    return _run_training(
        algorithm=Algorithm.REINFORCE,
        seed=seed,
        track_path=track_path,
        run_path=run_path,
        actor_config=resolved_actor_config,
        environment_config=environment_config,
        training_config=training_config,
        run_category=run_category,
        training_circuit_schedule=training_circuit_schedule,
        evaluation_circuits=evaluation_circuits,
        final_evaluation_circuits=final_evaluation_circuits,
        training_reference_circuits=training_reference_circuits,
        observation=observation,
        learning_rates={"actor_learning_rate": actor_learning_rate},
        agent_factory=lambda observation_dimensions, streams: ReinforceAgent(
            observation_dimensions=observation_dimensions,
            actor_config=resolved_actor_config,
            config=reinforce_config,
            initialization_generator=streams.get_torch_generator(
                SeedStream.ACTOR_INITIALIZATION,
                device=execution_config.device,
            ),
            sampling_generator=tuple(
                streams.get_torch_generator(
                    SeedStream.POLICY_ACTION_SAMPLING,
                    device=execution_config.device,
                    substream_identity=index,
                )
                for index in range(execution_config.environment_workers)
            ),
            device=execution_config.device,
            gradient_dispersion_subbatch=(
                training_config.logging.gradient_dispersion_subbatch
            ),
        ),
    )


def run_a2c_training(
    *,
    seed: int,
    track_path: str | Path | None = None,
    run_path: str | Path,
    actor_config: ActorConfig,
    actor_learning_rate: float,
    critic_learning_rate: float,
    training_interaction_budget: int,
    environment_config: EnvironmentConfig | None = None,
    a2c_config: A2CConfig | None = None,
    critic_config: CriticConfig = FIXED_CRITIC_CONFIG,
    evaluation_interval: int | None = None,
    near_saturated_steering_threshold: float | None = None,
    execution_config: ExecutionConfig | None = None,
    training_circuit_schedule: TrainingCircuitSchedule | None = None,
    evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    final_evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    training_reference_circuits: int = 0,
    observation: ObservationRepresentation = ObservationRepresentation.FRENET,
    run_category: RunCategory = RunCategory.REDUCED_VALIDATION,
) -> TrainingEngine:
    """
    Train A2C through the shared engine and persist its run records.
    """
    if training_interaction_budget <= 0:
        raise ValueError("Training interaction budget must be positive.")
    if actor_learning_rate <= 0 or critic_learning_rate <= 0:
        raise ValueError("A2C learning rates must be positive.")
    environment_config = environment_config or EnvironmentConfig()
    a2c_config = a2c_config or A2CConfig()
    execution_config = execution_config or ExecutionConfig()
    configure_torch_determinism(execution_config)
    resolved_actor_config = replace(actor_config, learning_rate=actor_learning_rate)
    base_training_config = TrainingConfig(
        actor=resolved_actor_config, critic=critic_config
    )
    resolved_evaluation_interval = (
        base_training_config.evaluation.evaluation_interval
        if evaluation_interval is None
        else evaluation_interval
    )
    training_config = replace(
        base_training_config,
        a2c=a2c_config,
        training_interaction_budget=training_interaction_budget,
        evaluation=replace(
            base_training_config.evaluation,
            evaluation_interval=resolved_evaluation_interval,
        ),
        logging=replace(
            base_training_config.logging,
            near_saturated_steering_threshold=(
                base_training_config.logging.near_saturated_steering_threshold
                if near_saturated_steering_threshold is None
                else near_saturated_steering_threshold
            ),
        ),
        execution=execution_config,
    )
    return _run_training(
        algorithm=Algorithm.A2C,
        seed=seed,
        track_path=track_path,
        run_path=run_path,
        actor_config=resolved_actor_config,
        environment_config=environment_config,
        training_config=training_config,
        run_category=run_category,
        training_circuit_schedule=training_circuit_schedule,
        evaluation_circuits=evaluation_circuits,
        final_evaluation_circuits=final_evaluation_circuits,
        training_reference_circuits=training_reference_circuits,
        observation=observation,
        learning_rates={
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
        },
        agent_factory=lambda observation_dimensions, streams: A2CAgent(
            observation_dimensions=observation_dimensions,
            actor_config=resolved_actor_config,
            critic_config=critic_config,
            config=a2c_config,
            critic_learning_rate=critic_learning_rate,
            actor_initialization_generator=streams.get_torch_generator(
                SeedStream.ACTOR_INITIALIZATION,
                device=execution_config.device,
            ),
            critic_initialization_generator=streams.get_torch_generator(
                SeedStream.CRITIC_INITIALIZATION,
                device=execution_config.device,
            ),
            sampling_generator=tuple(
                streams.get_torch_generator(
                    SeedStream.POLICY_ACTION_SAMPLING,
                    device=execution_config.device,
                    substream_identity=index,
                )
                for index in range(execution_config.environment_workers)
            ),
            device=execution_config.device,
            gradient_dispersion_subbatch=(
                training_config.logging.gradient_dispersion_subbatch
            ),
        ),
    )


def run_ppo_training(
    *,
    seed: int,
    track_path: str | Path | None = None,
    run_path: str | Path,
    actor_config: ActorConfig,
    actor_learning_rate: float,
    critic_learning_rate: float,
    training_interaction_budget: int,
    environment_config: EnvironmentConfig | None = None,
    ppo_config: PPOConfig | None = None,
    critic_config: CriticConfig = FIXED_CRITIC_CONFIG,
    evaluation_interval: int | None = None,
    near_saturated_steering_threshold: float | None = None,
    execution_config: ExecutionConfig | None = None,
    training_circuit_schedule: TrainingCircuitSchedule | None = None,
    evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    final_evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    training_reference_circuits: int = 0,
    observation: ObservationRepresentation = ObservationRepresentation.FRENET,
    run_category: RunCategory = RunCategory.REDUCED_VALIDATION,
) -> TrainingEngine:
    """
    Train PPO through the shared engine and persist its run records.
    """
    if training_interaction_budget <= 0:
        raise ValueError("Training interaction budget must be positive.")
    if actor_learning_rate <= 0 or critic_learning_rate <= 0:
        raise ValueError("PPO learning rates must be positive.")
    environment_config = environment_config or EnvironmentConfig()
    ppo_config = ppo_config or PPOConfig()
    execution_config = execution_config or ExecutionConfig()
    configure_torch_determinism(execution_config)
    resolved_actor_config = replace(actor_config, learning_rate=actor_learning_rate)
    base_training_config = TrainingConfig(
        actor=resolved_actor_config, critic=critic_config
    )
    resolved_evaluation_interval = (
        base_training_config.evaluation.evaluation_interval
        if evaluation_interval is None
        else evaluation_interval
    )
    training_config = replace(
        base_training_config,
        ppo=ppo_config,
        training_interaction_budget=training_interaction_budget,
        evaluation=replace(
            base_training_config.evaluation,
            evaluation_interval=resolved_evaluation_interval,
        ),
        logging=replace(
            base_training_config.logging,
            near_saturated_steering_threshold=(
                base_training_config.logging.near_saturated_steering_threshold
                if near_saturated_steering_threshold is None
                else near_saturated_steering_threshold
            ),
        ),
        execution=execution_config,
    )
    return _run_training(
        algorithm=Algorithm.PPO,
        seed=seed,
        track_path=track_path,
        run_path=run_path,
        actor_config=resolved_actor_config,
        environment_config=environment_config,
        training_config=training_config,
        run_category=run_category,
        training_circuit_schedule=training_circuit_schedule,
        evaluation_circuits=evaluation_circuits,
        final_evaluation_circuits=final_evaluation_circuits,
        training_reference_circuits=training_reference_circuits,
        observation=observation,
        learning_rates={
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
        },
        agent_factory=lambda observation_dimensions, streams: PPOAgent(
            observation_dimensions=observation_dimensions,
            actor_config=resolved_actor_config,
            critic_config=critic_config,
            config=ppo_config,
            critic_learning_rate=critic_learning_rate,
            actor_initialization_generator=streams.get_torch_generator(
                SeedStream.ACTOR_INITIALIZATION,
                device=execution_config.device,
            ),
            critic_initialization_generator=streams.get_torch_generator(
                SeedStream.CRITIC_INITIALIZATION,
                device=execution_config.device,
            ),
            sampling_generator=tuple(
                streams.get_torch_generator(
                    SeedStream.POLICY_ACTION_SAMPLING,
                    device=execution_config.device,
                    substream_identity=index,
                )
                for index in range(execution_config.environment_workers)
            ),
            optimization_generator=streams.get_torch_generator(
                SeedStream.OPTIMIZATION_BATCH_ORDER,
                device=execution_config.device,
            ),
            device=execution_config.device,
            gradient_dispersion_subbatch=(
                training_config.logging.gradient_dispersion_subbatch
            ),
        ),
    )


def _run_training(
    *,
    algorithm: Algorithm,
    seed: int,
    track_path: str | Path | None = None,
    run_path: str | Path,
    actor_config: ActorConfig,
    environment_config: EnvironmentConfig,
    training_config: TrainingConfig,
    run_category: RunCategory,
    training_circuit_schedule: TrainingCircuitSchedule | None = None,
    evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    final_evaluation_circuits: Sequence[EvaluationCircuit] | None = None,
    training_reference_circuits: int = 0,
    observation: ObservationRepresentation = ObservationRepresentation.FRENET,
    learning_rates: dict[str, float],
    agent_factory: Callable[[int, RunSeedStreams], ParameterizedOnPolicyAgent],
) -> TrainingEngine:
    """
    Train one selected on-policy agent through the common run lifecycle.
    """
    execution_config = training_config.execution
    steering_threshold = training_config.logging.near_saturated_steering_threshold
    if run_category is RunCategory.REPORTED and steering_threshold is None:
        raise ValueError(
            "Reported runs require an explicit near-saturated steering threshold."
        )
    streams = RunSeedStreams(_seed_namespace(run_category), seed)
    # The representation is a property of the run, so it is applied to the
    # configuration that every environment and the stored config document share.
    environment_config = replace(environment_config, observation_type=observation)
    track = _prototype_track(
        track_path,
        training_circuit_schedule=training_circuit_schedule,
        environment_config=environment_config,
    )
    run = RunDirectory.create(
        run_path,
        category=run_category,
        # Every dimension the design matrices vary appears here. Experiment 2
        # runs two observations at one algorithm, actor and root, so leaving the
        # observation out gave its paired runs the same identity, and analysis
        # tables keyed by run identity kept only whichever loaded last.
        run_id=(
            f"{algorithm.value}-{actor_config.name}-{observation.value}-seed-{seed}"
        ),
        manifest={
            "purpose": f"{algorithm.value}_training",
            "algorithm": algorithm.value,
            "root_seed": seed,
            "seed_namespace": streams.namespace.name,
            "track_path": None if track_path is None else str(track_path),
            "track_seed": track.track.generation.seed,
            "observation_type": observation.value,
            # A multi-circuit run has no single circuit, so what identifies its
            # geometry is the schedule it draws from and the splits it is
            # measured on. The track seed above is only the prototype's.
            "training_circuit_schedule": (
                None
                if training_circuit_schedule is None
                else training_circuit_schedule.namespace.name
            ),
            "evaluation_circuits": _evaluation_circuit_manifest(evaluation_circuits),
            "seed_streams": {
                stream.name: _first_seed(streams, stream) for stream in SeedStream
            },
            "worker_seed_streams": {
                str(index): {
                    stream.name: _first_indexed_seed(streams, stream, index)
                    for stream in (
                        SeedStream.POLICY_ACTION_SAMPLING,
                        SeedStream.ENVIRONMENT_RESETS,
                        SeedStream.TRAINING_TRACK_SELECTION,
                    )
                }
                for index in range(execution_config.environment_workers)
            },
        },
        config={
            "training": training_config.to_dict(),
            "environment": environment_config.to_dict(),
            **learning_rates,
        },
        metadata=collect_run_metadata(
            repository=Path(__file__).resolve().parents[1],
            device=execution_config.device,
            environment_workers=execution_config.environment_workers,
            intraop_threads=execution_config.intraop_threads,
            interop_threads=execution_config.interop_threads,
        ),
    )
    # Training samples a start pose for coverage, but deterministic evaluation
    # must always launch from the canonical start line: a reported evaluation
    # curve has to answer the same question at every checkpoint.
    evaluation_config = replace(
        environment_config, start=StartStateConfig(randomized=False)
    )
    observation_shape = observation_space_for(track, environment_config).shape
    if observation_shape is None or observation_shape[0] is None:
        raise ValueError("RacingEnv must expose a fixed observation dimension.")
    observation_dimensions = observation_shape[0]
    agent = agent_factory(observation_dimensions, streams)
    engine = _ENGINES[algorithm](
        agent,
        track,
        environment_config,
        RunningObservationNormalizer(
            observation_dimensions, training_config.normalization
        ),
        run_category=run_category,
        evaluation_environment_factory=(
            None
            if evaluation_circuits is not None
            else lambda: RacingEnv(track, config=evaluation_config)
        ),
        evaluation_circuits=evaluation_circuits,
        evaluation_interval=training_config.evaluation.evaluation_interval,
        environment_reset_generators=tuple(
            streams.get_numpy_generator(
                SeedStream.ENVIRONMENT_RESETS,
                substream_identity=index,
            )
            for index in range(execution_config.environment_workers)
        ),
        track_selection_generators=tuple(
            streams.get_numpy_generator(
                SeedStream.TRAINING_TRACK_SELECTION,
                substream_identity=index,
            )
            for index in range(execution_config.environment_workers)
        ),
        training_circuit_schedule=training_circuit_schedule,
        execution_config=execution_config,
        evaluation_seed=_first_seed(streams, SeedStream.EVALUATION),
        root_identity=seed,
        circuit_identity=str(track.track.generation.seed),
        circuit_split=CircuitSplit.DEVELOPMENT.value,
        observation_type=observation,
        near_saturated_steering_threshold=steering_threshold,
    )
    started = perf_counter()
    try:
        _train_with_checkpoints(engine, training_config, run.path / "checkpoints")
        engine.save(run.path / "checkpoints" / "final.pt")
        if training_reference_circuits:
            # An in-sample reference: circuits this run trained on, revisited
            # deterministically so the gap to held-out circuits is measurable.
            engine.evaluate(
                engine.training_reference_circuits(training_reference_circuits)
            )
        if final_evaluation_circuits:
            # Held-out circuits are opened here and nowhere else: after the last
            # optimizer step and the saved final policy, so nothing they reveal
            # can reach training, checkpoint selection or the recorded curve.
            engine.evaluate(final_evaluation_circuits)
        persistence_started = perf_counter()
        _write_engine_records(
            run,
            engine,
            trajectory_interval=training_config.logging.trajectory_interval,
            trajectory_circuits_per_boundary=(
                training_config.logging.trajectory_circuits_per_boundary
            ),
            final_interactions=training_config.training_interaction_budget,
        )
        record_persistence = perf_counter() - persistence_started
        state = engine.state()
        resources = ResourceRecord(
            run_category=run_category,
            scope=MetricScope.TRAINING,
            training_interactions=state.counters.training_interactions,
            evaluation_interactions=state.counters.evaluation_interactions,
            completed_episodes=sum(
                record.outcome.value == "completed" for record in engine.episode_records
            ),
            optimizer_updates=state.counters.optimizer_updates,
            actor_parameters=agent.actor_parameter_count,
            critic_parameters=agent.critic_parameter_count,
            peak_process_memory=None,
        )
        run.complete(
            {
                "training_interactions": state.counters.training_interactions,
                "evaluation_interactions": state.counters.evaluation_interactions,
                "finished_episodes": state.counters.finished_episodes,
                "optimizer_updates": state.counters.optimizer_updates,
                "actor_parameters": agent.actor_parameter_count,
                "critic_parameters": agent.critic_parameter_count,
                "resources": resources.to_dict(),
                "timing": TimingRecord(
                    run_category=run_category,
                    scope=MetricScope.TRAINING,
                    collection=state.timing.collection,
                    optimization=state.timing.optimization,
                    evaluation=state.timing.evaluation,
                    persistence=state.timing.persistence + record_persistence,
                    end_to_end=perf_counter() - started,
                ).to_dict(),
            }
        )
    finally:
        engine.close()
    return engine


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse the explicit algorithm settings and isolated run output location.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=tuple(algorithm.value for algorithm in Algorithm),
        default=Algorithm.REINFORCE.value,
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--track", required=True)
    parser.add_argument("--run-path", "--output", dest="run_path", required=True)
    parser.add_argument("--actor-size", choices=tuple(_ACTORS), default="medium")
    parser.add_argument("--actor-learning-rate", required=True, type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--interaction-budget", required=True, type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--near-saturated-steering-threshold", type=float)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--run-category",
        choices=tuple(category.value for category in RunCategory),
        default=RunCategory.REDUCED_VALIDATION.value,
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """
    Train the selected agent and print its exact completed interaction count.
    """
    parsed = parse_arguments(arguments)
    algorithm = Algorithm(parsed.algorithm)
    worker_count = parsed.num_envs
    execution_config = ExecutionConfig()
    if worker_count is not None:
        execution_config = replace(execution_config, environment_workers=worker_count)
    common_arguments = {
        "seed": parsed.seed,
        "track_path": parsed.track,
        "run_path": parsed.run_path,
        "actor_config": _ACTORS[parsed.actor_size],
        "actor_learning_rate": parsed.actor_learning_rate,
        "training_interaction_budget": parsed.interaction_budget,
        "evaluation_interval": parsed.evaluation_interval,
        "near_saturated_steering_threshold": (parsed.near_saturated_steering_threshold),
        "execution_config": execution_config,
        "run_category": RunCategory(parsed.run_category),
    }
    if algorithm is Algorithm.REINFORCE:
        engine = run_reinforce_training(**common_arguments)
    else:
        if parsed.critic_learning_rate is None:
            raise ValueError(
                f"{algorithm.value.upper()} requires --critic-learning-rate."
            )
        if algorithm is Algorithm.A2C:
            engine = run_a2c_training(
                **common_arguments, critic_learning_rate=parsed.critic_learning_rate
            )
        else:
            engine = run_ppo_training(
                **common_arguments, critic_learning_rate=parsed.critic_learning_rate
            )
    print(
        f"{parsed.algorithm.upper()} completed "
        f"{engine.state().counters.training_interactions} training interactions."
    )
    return 0


def _write_engine_records(
    run: RunDirectory,
    engine: TrainingEngine,
    *,
    trajectory_interval: int,
    trajectory_circuits_per_boundary: int,
    final_interactions: int,
) -> None:
    """
    Persist shared episode/evaluation records and algorithm diagnostics.
    """
    for record in engine.episode_records:
        run.append("episodes", record)
    retained_at_boundary: Counter[int] = Counter()
    for evaluation in engine.evaluations:
        run.append("evaluations", evaluation.record)
        boundary = evaluation.record.training_interactions
        qualifies = (
            boundary == final_interactions or boundary % trajectory_interval == 0
        )
        if qualifies and (
            retained_at_boundary[boundary] < trajectory_circuits_per_boundary
        ):
            retained_at_boundary[boundary] += 1
            run.write_trajectory(
                f"evaluation_{evaluation.record.evaluation_index}_interaction_{boundary}",
                {
                    "evaluation": evaluation.record.to_dict(),
                    "transitions": [
                        transition.to_dict() for transition in evaluation.transitions
                    ],
                },
            )
    for update in engine.updates:
        diagnostics = update.output.diagnostics
        run.append(
            "updates",
            UpdateRecord(
                run_category=run.category,
                update_index=update.update_index,
                training_interactions=update.training_interactions,
                actor_loss=float(_diagnostic(diagnostics, "actor_loss")),
                critic_loss=_optional_diagnostic(diagnostics, "critic_loss"),
                actor_gradient_norm=float(
                    _diagnostic(diagnostics, "actor_gradient_norm")
                ),
                critic_gradient_norm=_optional_diagnostic(
                    diagnostics, "critic_gradient_norm"
                ),
                optimization_duration=update.optimization_duration,
                actor_learning_rate=float(
                    _diagnostic(diagnostics, "actor_learning_rate")
                ),
                entropy_proxy=float(_diagnostic(diagnostics, "entropy_proxy")),
                log_standard_deviation=(
                    float(_diagnostic(diagnostics, "log_standard_deviation_0")),
                    float(_diagnostic(diagnostics, "log_standard_deviation_1")),
                ),
                actor_weight_norm=float(_diagnostic(diagnostics, "actor_weight_norm")),
                actor_update_norm=float(_diagnostic(diagnostics, "actor_update_norm")),
                critic_learning_rate=_optional_diagnostic(
                    diagnostics, "critic_learning_rate"
                ),
                critic_weight_norm=_optional_diagnostic(
                    diagnostics, "critic_weight_norm"
                ),
                critic_update_norm=_optional_diagnostic(
                    diagnostics, "critic_update_norm"
                ),
                explained_variance=_optional_diagnostic(
                    diagnostics, "explained_variance"
                ),
                approximate_kl=_optional_diagnostic(diagnostics, "approximate_kl"),
                clip_fraction=_optional_diagnostic(diagnostics, "clip_fraction"),
                diagnostics={
                    key: value
                    for key, value in diagnostics.items()
                    if key
                    not in {
                        "actor_loss",
                        "actor_gradient_norm",
                        "actor_weight_norm",
                        "actor_update_norm",
                        "critic_loss",
                        "critic_gradient_norm",
                        "critic_learning_rate",
                        "critic_weight_norm",
                        "critic_update_norm",
                        "explained_variance",
                        "approximate_kl",
                        "clip_fraction",
                        "actor_learning_rate",
                        "entropy_proxy",
                        "log_standard_deviation_0",
                        "log_standard_deviation_1",
                    }
                },
            ),
        )


def _train_with_checkpoints(
    engine: TrainingEngine,
    config: TrainingConfig,
    checkpoint_directory: Path,
) -> None:
    """
    Train to each configured checkpoint boundary without finalizing partial batches.
    """
    target = min(config.checkpoint_interval, config.training_interaction_budget)
    while target < config.training_interaction_budget:
        engine.train(target, finalize=False)
        engine.save(checkpoint_directory / f"interaction_{target}.pt")
        target = min(
            target + config.checkpoint_interval,
            config.training_interaction_budget,
        )
    engine.train(config.training_interaction_budget)


def _prototype_track(
    track_path: str | Path | None,
    *,
    training_circuit_schedule: TrainingCircuitSchedule | None,
    environment_config: EnvironmentConfig,
) -> TrackWithGeometry:
    """
    Return the circuit the run is configured around, or one standing in for many.

    A scheduled run replaces its circuit at every reset, so it has no saved
    circuit to load. It still needs one prepared track to fix the observation
    space and to give the workers something to hold before their first reset.
    That circuit is the schedule's own first identity, and no recorded outcome
    is attributed to it.
    """
    if track_path is not None:
        return TrackWithGeometry.load(
            track_path,
            vehicle_config=environment_config.vehicle,
            track_config=environment_config.track,
        )
    if training_circuit_schedule is None:
        raise ValueError("Training requires a saved circuit or a circuit schedule.")
    return TrackWithGeometry.generate(
        circuit_track_seed(training_circuit_schedule.namespace, 0),
        track_config=environment_config.track,
        vehicle_config=environment_config.vehicle,
    )


def _evaluation_circuit_manifest(
    circuits: Sequence[EvaluationCircuit] | None,
) -> dict[str, list[str]] | None:
    """
    Record which circuits, by split, a run was measured on.
    """
    if circuits is None:
        return None
    manifest: dict[str, list[str]] = {}
    for circuit in circuits:
        split = "unassigned" if circuit.split is None else circuit.split.value
        manifest.setdefault(split, []).append(circuit.identity)
    return manifest


def _seed_namespace(category: RunCategory) -> SeedNamespace:
    """
    Select the documented seed namespace corresponding to one run category.
    """
    if category is RunCategory.PRE_EXPERIMENT:
        return SeedNamespace.LEARNING_RATE_CALIBRATION
    if category is RunCategory.REPORTED:
        return SeedNamespace.EXPERIMENT_1_REPORTED
    return SeedNamespace.REDUCED_BUDGET_VALIDATION


def _diagnostic(diagnostics: dict[str, float | int | None], key: str) -> float | int:
    """
    Return one required shared diagnostic with an explicit absent-value error.
    """
    value = diagnostics.get(key)
    if value is None:
        raise ValueError(f"Update lacks required diagnostic: {key}.")
    return value


def _optional_diagnostic(
    diagnostics: dict[str, float | int | None], key: str
) -> float | int | None:
    """
    Return one algorithm-specific diagnostic when the active agent records it.
    """
    return diagnostics.get(key)


def _first_seed(streams: RunSeedStreams, stream: SeedStream) -> int:
    """
    Return one reproducible integer identifier for a named seed stream.
    """
    return int(streams.get_numpy_generator(stream).integers(0, 2**32))


def _first_indexed_seed(
    streams: RunSeedStreams,
    stream: SeedStream,
    substream_identity: int,
) -> int:
    """
    Return one reproducible identifier for a worker-specific seed substream.
    """
    return int(
        streams.get_numpy_generator(
            stream,
            substream_identity=substream_identity,
        ).integers(0, 2**32)
    )


if __name__ == "__main__":
    raise SystemExit(main())
