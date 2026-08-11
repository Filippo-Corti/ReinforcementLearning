"""Persistent process-based vector collection for racing environments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from os import getpid
from typing import Any, Self

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from numpy.typing import NDArray

from configs import EnvironmentConfig, ExecutionConfig
from envs.racing import RacingEnv, RacingEnvState
from envs.tracks import TrackWithGeometry
from utils.random import TorchDeterminismState, configure_torch_determinism

_OPTIONAL_INTEGER_INFO_FIELDS = frozenset({"collision_substep"})


def vector_info(infos: dict[str, Any], key: str, environment_index: int) -> Any:
    """
    Return one worker's value from Gymnasium's dictionary-of-arrays info.
    """
    availability = infos.get(f"_{key}")
    if availability is not None and not bool(availability[environment_index]):
        raise KeyError(f"Worker {environment_index} did not provide info field {key}.")
    value = infos[key][environment_index]
    if key in _OPTIONAL_INTEGER_INFO_FIELDS and int(value) == -1:
        return None
    return value


def vector_worker_info(infos: dict[str, Any], environment_index: int) -> dict[str, Any]:
    """
    Reconstruct one worker's semantic info mapping from vectorized output.
    """
    worker_info = {
        key: values[environment_index]
        for key, values in infos.items()
        if not key.startswith("_")
        and (f"_{key}" not in infos or bool(infos[f"_{key}"][environment_index]))
    }
    for key in _OPTIONAL_INTEGER_INFO_FIELDS:
        if key in worker_info and int(worker_info[key]) == -1:
            worker_info[key] = None
    return worker_info


@dataclass(frozen=True, slots=True)
class RacingWorkerState:
    """
    Store one worker's selected track and resumable environment state.

    Fields:
        * track_index: Index of the current track in the immutable worker pool.
        * environment: Racing dynamics and lifecycle state.
        * last_observation: Observation returned by the most recent reset or step.
        * last_info: Diagnostics paired with the most recent observation.
    """

    track_index: int
    environment: RacingEnvState
    last_observation: NDArray[np.float32]
    last_info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorRacingState:
    """
    Store vector-worker and per-worker scheduling RNG state for exact resume.

    Fields:
        * workers: Ordered environment snapshots, one per persistent process.
        * reset_generators: Ordered NumPy bit-generator states for reset seeds.
        * track_generators: Ordered NumPy bit-generator states for track selection.
        * next_track_indices: Track indices last assigned to each worker.
    """

    workers: tuple[RacingWorkerState, ...]
    reset_generators: tuple[dict[str, Any], ...]
    track_generators: tuple[dict[str, Any], ...]
    next_track_indices: tuple[int, ...]


class _RacingWorkerEnv(gym.Env[NDArray[np.float32], dict[str, Any]]):
    """
    Own one replaceable racing environment inside a persistent worker process.

    The additional `enabled` action flag lets the parent leave a finished worker
    parked while other workers continue. Parked calls do not advance dynamics and
    are marked invalid so they never enter a learning buffer.

    Fields:
        * tracks: Immutable circuits available to this worker.
        * environment_config: Shared racing configuration.
        * next_track_index: Circuit selected by the parent for the next reset.
        * torch_determinism: Deterministic PyTorch settings applied in this process.
    """

    metadata = {"autoreset_mode": AutoresetMode.DISABLED.value}  # noqa: RUF012

    def __init__(
        self,
        tracks: tuple[TrackWithGeometry, ...],
        environment_config: EnvironmentConfig,
        torch_determinism: TorchDeterminismState,
    ) -> None:
        super().__init__()
        self.tracks = tracks
        self.environment_config = environment_config
        self.next_track_index = 0
        self.torch_determinism = torch_determinism
        self._track_index = 0
        self._environment = RacingEnv(tracks[0], config=environment_config)
        self._last_observation = np.zeros(4, dtype=np.float32)
        self._last_info: dict[str, Any] = {}
        self.observation_space = self._environment.observation_space
        self.action_space = gym.spaces.Dict(
            {
                "action": self._environment.action_space,
                "enabled": gym.spaces.Discrete(2),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Replace the selected circuit and reset only this worker's episode.
        """
        super().reset(seed=seed)
        del options
        self._environment.close()
        self._track_index = int(self.next_track_index)
        self._environment = RacingEnv(
            self.tracks[self._track_index],
            config=self.environment_config,
        )
        observation, info = self._environment.reset(seed=seed)
        self._last_observation = np.asarray(observation, dtype=np.float32).copy()
        self._last_info = self._worker_info(info, transition_valid=False)
        return self._last_observation.copy(), dict(self._last_info)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Advance one enabled worker or return an explicitly invalid parked row.
        """
        if not bool(action["enabled"]):
            info = dict(self._last_info)
            info["transition_valid"] = False
            return self._last_observation.copy(), 0.0, False, False, info

        observation, reward, terminated, truncated, info = self._environment.step(
            np.asarray(action["action"], dtype=np.float32)
        )
        self._last_observation = np.asarray(observation, dtype=np.float32).copy()
        self._last_info = self._worker_info(info, transition_valid=True)
        return (
            self._last_observation.copy(),
            float(reward),
            terminated,
            truncated,
            dict(self._last_info),
        )

    def close(self) -> None:
        """
        Release resources owned by the current racing environment.
        """
        self._environment.close()

    @property
    def worker_pid(self) -> int:
        """
        Return the persistent operating-system process identity.
        """
        return getpid()

    @property
    def worker_state(self) -> RacingWorkerState:
        """
        Return the semantic state needed to resume this worker exactly.
        """
        return RacingWorkerState(
            track_index=self._track_index,
            environment=self._environment.snapshot(),
            last_observation=self._last_observation.copy(),
            last_info=deepcopy(self._last_info),
        )

    @worker_state.setter
    def worker_state(self, state: RacingWorkerState) -> None:
        self._environment.close()
        self._track_index = state.track_index
        self.next_track_index = state.track_index
        self._environment = RacingEnv(
            self.tracks[self._track_index],
            config=self.environment_config,
        )
        self._environment.restore(state.environment)
        self._last_observation = np.asarray(
            state.last_observation, dtype=np.float32
        ).copy()
        self._last_info = deepcopy(state.last_info)

    def _worker_info(
        self,
        info: dict[str, Any],
        *,
        transition_valid: bool,
    ) -> dict[str, Any]:
        enriched = dict(info)
        for key in _OPTIONAL_INTEGER_INFO_FIELDS:
            if enriched.get(key) is None:
                # Gymnasium merges worker infos into one homogeneous NumPy array.
                enriched[key] = -1
        track = self.tracks[self._track_index].track
        enriched.update(
            {
                "transition_valid": transition_valid,
                "track_index": self._track_index,
                "track_length": track.track_length,
                "circuit_identity": str(track.generation.seed),
            }
        )
        return enriched


def _make_racing_worker(
    tracks: tuple[TrackWithGeometry, ...],
    environment_config: EnvironmentConfig,
    execution_config: ExecutionConfig,
) -> _RacingWorkerEnv:
    """
    Configure deterministic CPU PyTorch settings inside one spawned worker.
    """
    worker_execution = replace(execution_config, device="cpu")
    torch_determinism = configure_torch_determinism(worker_execution)
    return _RacingWorkerEnv(tracks, environment_config, torch_determinism)


class PersistentRacingVectorEnv:
    """
    Step racing environments in persistent spawned CPU worker processes.

    Each worker owns independent reset and track-selection generators. Workers
    are spawned once at construction, can be reset individually after episode
    boundaries, and remain alive until `close`.

    Fields:
        * num_envs: Number of persistent environment processes.
        * tracks: Immutable circuits shared by worker construction.
        * reset_generators: Independent reset-seed streams by worker index.
        * track_generators: Independent circuit-selection streams by worker index.
    """

    def __init__(
        self,
        tracks: tuple[TrackWithGeometry, ...],
        environment_config: EnvironmentConfig,
        execution_config: ExecutionConfig,
        reset_generators: tuple[np.random.Generator, ...],
        track_generators: tuple[np.random.Generator, ...],
    ) -> None:
        """
        Spawn the configured worker processes once and retain their pipes.
        """
        if not tracks:
            raise ValueError("Vector racing collection requires at least one circuit.")
        if execution_config.environment_workers <= 0:
            raise ValueError(
                "Vector racing collection requires a positive worker count."
            )
        self.num_envs = execution_config.environment_workers
        if len(reset_generators) != self.num_envs:
            raise ValueError("One reset generator is required per environment worker.")
        if len(track_generators) != self.num_envs:
            raise ValueError("One track generator is required per environment worker.")
        self.tracks = tracks
        self.reset_generators = reset_generators
        self.track_generators = track_generators
        self._next_track_indices = np.zeros(self.num_envs, dtype=np.int64)
        environment_functions = tuple(
            partial(
                _make_racing_worker,
                tracks,
                environment_config,
                execution_config,
            )
            for _ in range(self.num_envs)
        )
        self._vector = AsyncVectorEnv(
            environment_functions,
            context="spawn",
            shared_memory=True,
            autoreset_mode=AutoresetMode.DISABLED,
        )

    def reset(
        self,
        mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Reset all workers or only those selected by a boolean mask.
        """
        reset_mask = (
            np.ones(self.num_envs, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if reset_mask.shape != (self.num_envs,):
            raise ValueError("The reset mask must contain one value per worker.")
        if not np.any(reset_mask):
            raise ValueError("At least one worker must be selected for reset.")

        seeds: list[int | None] = [None] * self.num_envs
        for environment_index in np.flatnonzero(reset_mask):
            index = int(environment_index)
            self._next_track_indices[index] = int(
                self.track_generators[index].integers(0, len(self.tracks))
            )
            seeds[index] = int(
                self.reset_generators[index].integers(
                    0,
                    2**32,
                    dtype=np.uint32,
                )
            )
        self._vector.set_attr(
            "next_track_index",
            [int(value) for value in self._next_track_indices],
        )
        options = None if mask is None else {"reset_mask": reset_mask.copy()}
        observations, infos = self._vector.reset(seed=seeds, options=options)
        return np.asarray(observations, dtype=np.float32), infos

    def step(
        self,
        actions: NDArray[np.float32],
        active: NDArray[np.bool_] | None = None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """
        Step enabled workers concurrently while leaving parked workers unchanged.
        """
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.num_envs, 2):
            raise ValueError("Actions must have shape (num_envs, 2).")
        active_mask = (
            np.ones(self.num_envs, dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        if active_mask.shape != (self.num_envs,):
            raise ValueError("The active mask must contain one value per worker.")
        observations, rewards, terminated, truncated, infos = self._vector.step(
            {
                "action": action_array,
                "enabled": active_mask.astype(np.int64),
            }
        )
        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            infos,
        )

    def state(self) -> VectorRacingState:
        """
        Return worker snapshots and independent scheduler-generator states.
        """
        workers = self._vector.get_attr("worker_state")
        return VectorRacingState(
            workers=tuple(workers),
            reset_generators=tuple(
                dict(deepcopy(generator.bit_generator.state))
                for generator in self.reset_generators
            ),
            track_generators=tuple(
                dict(deepcopy(generator.bit_generator.state))
                for generator in self.track_generators
            ),
            next_track_indices=tuple(int(value) for value in self._next_track_indices),
        )

    def restore(self, state: VectorRacingState) -> None:
        """
        Restore every worker and scheduling generator without respawning.
        """
        if len(state.workers) != self.num_envs:
            raise ValueError("Vector state worker count does not match this pool.")
        for generator, generator_state in zip(
            self.reset_generators, state.reset_generators, strict=True
        ):
            generator.bit_generator.state = deepcopy(generator_state)
        for generator, generator_state in zip(
            self.track_generators, state.track_generators, strict=True
        ):
            generator.bit_generator.state = deepcopy(generator_state)
        self._next_track_indices = np.asarray(
            state.next_track_indices, dtype=np.int64
        ).copy()
        self._vector.set_attr("worker_state", state.workers)

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """
        Return process identities for persistence checks and run metadata.
        """
        return tuple(int(value) for value in self._vector.get_attr("worker_pid"))

    @property
    def worker_torch_determinism(self) -> tuple[TorchDeterminismState, ...]:
        """
        Return deterministic PyTorch settings observed inside each worker.
        """
        return tuple(self._vector.get_attr("torch_determinism"))

    def close(self) -> None:
        """
        Close every persistent worker process.
        """
        self._vector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
