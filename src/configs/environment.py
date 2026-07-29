"""Typed and validated configuration for the racing environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Configuration for simulation time.

    Fields:
        * agent_timestep: The time unit at which the agent operates, in seconds.
        * physics_timestep: The time unit at which the physics simulation operates, in seconds.
        * physics_substeps: The number of physics substeps per agent step (agent_timestep / physics_timestep).
        * max_episode_steps: The maximum number of timesteps per episode, after which the episode will terminate
    """

    agent_timestep: float = 0.04
    physics_timestep: float = 0.01
    physics_substeps: int = 4
    max_episode_steps: int = 5_000


@dataclass(frozen=True, slots=True)
class CarConfig:
    """
    Physical configuration of the racing car.

    Fields:
        * wheelbase_m: The distance between the front and rear axles of the vehicle, in meters.
        * max_acceleration_m_per_s2: The maximum acceleration of the vehicle, in m/s^2.
        * max_steering_angle_deg: The maximum steering angle of the vehicle, in degrees.
        * max_speed_m_per_s: The maximum speed of the vehicle, in m/s.
    """

    wheelbase_m: float = 3.6
    max_acceleration: float = 9.26
    max_steering_angle: float = 30.0
    max_speed: float = 70.0


@dataclass(frozen=True, slots=True)
class TrackGenerationConfig:
    """
    Configuration for the procedural track generation validation.
    The track is generated starting from a circle of radius ``base_radius_m``
    and then jittered radially and angularly to create a smooth, closed track.

    Fields:
        * n_checkpoints: The number of checkpoints used to define the track's shape. Must be at least 3.
        * base_radius_m: The base radius of the track, in meters.
        * radial_jitter: The fraction of the base radius by which checkpoints can vary radially.
        * angular_jitter: The fraction of the angular range by which checkpoints can vary.
        * sample_spacing: The spacing between samples along the track, in meters.
        * width: The width of the track, in meters.
        * max_attempts: The maximum number of attempts to generate a valid track.
        * min_length: The minimum length of the track, in meters.
        * max_length: The maximum length of the track, in meters.
        * nonlocal_centerline_margin: The margin around the centerline that must be free of obstacles, in meters.
    """

    n_checkpoints: int = 12
    base_radius: float = 250.0
    radial_jitter: float = 0.25
    angular_jitter: float = 0.25
    sample_spacing: float = 0.5
    width: float = 12.0
    max_attempts: int = 100
    min_length: float = 1_000.0
    max_length: float = 3_000.0
    nonlocal_centerline_margin: float = 2.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Configuration of the reward function.

    Fields:
        * finish_reward: The reward given to the agent for successfully completing the track.
        * crash_penalty: The penalty given to the agent for crashing into an obstacle or going off-track.
        * time_penalty_rate: The penalty rate applied to the agent's reward for each second of simulation time.
        * progress_coefficient: The coefficient applied to the agent's reward based on its progress along the track.
    """

    finish_reward: float = 10.0
    crash_penalty: float = 20.0
    time_penalty_rate: float = 0.05
    progress_coefficient: float = 1.0


@dataclass(frozen=True, slots=True)
class FrenetObservationConfig:
    """Configuration of the Frenet Observation of the environment.
    
    Fields:
        * lookahead_base: The base lookahead distance for the Frenet observation, in meters.
        * lookahead_speed_factor: The speed factor used to adjust the lookahead distance based on the speed, in seconds.
    """

    lookahead_base: float =  5.0 
    lookahead_speed_factor: float = 0.7  


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Complete configuration of environment-owned behaviour."""
    
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    vehicle: CarConfig = field(default_factory=CarConfig)
    track: TrackGenerationConfig = field(default_factory=TrackGenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: FrenetObservationConfig = field(
        default_factory=FrenetObservationConfig
    )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic plain-dictionary representation."""
        return asdict(self)
