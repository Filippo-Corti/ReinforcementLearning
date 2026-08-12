"""Configuration for the racing environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """
    Complete configuration of environment-owned behaviour.

    Fields:
        * simulation: Simulation timing and episode settings.
        * vehicle: Physical car settings.
        * track: Track generation and validation settings.
        * reward: Reward function settings.
        * observation: Frenet observation settings.
        * start: Episode start-pose sampling settings.
    """

    simulation: SimulationConfig = field(default_factory=lambda: SimulationConfig())
    vehicle: CarConfig = field(default_factory=lambda: CarConfig())
    track: TrackGenerationConfig = field(
        default_factory=lambda: TrackGenerationConfig()
    )
    reward: RewardConfig = field(default_factory=lambda: RewardConfig())
    observation: FrenetObservationConfig = field(
        default_factory=lambda: FrenetObservationConfig()
    )
    start: StartStateConfig = field(default_factory=lambda: StartStateConfig())

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deterministic plain-dictionary representation.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Configuration for simulation time.

    Fields:
        * agent_timestep: The time unit at which the agent operates, in seconds.
        * physics_timestep: The time unit at which the physics simulation operates, in seconds.
        * physics_substeps: The number of physics substeps per agent step (agent_timestep / physics_timestep).
        * max_episode_steps: The maximum number of timesteps per episode, after which the episode will terminate
        * stall_time: The window over which a car must make progress, in seconds.
        * stall_progress: The progress required within that window, in meters.
    """

    agent_timestep: float = 0.04
    physics_timestep: float = 0.01
    physics_substeps: int = 4
    max_episode_steps: int = 1_000
    stall_time: float = 3.0
    stall_progress: float = 1.0


@dataclass(frozen=True, slots=True)
class StartStateConfig:
    """
    Configuration of the pose an episode starts from.

    A single fixed start pose makes the zero-speed launch the only state the
    agent ever has to solve, and leaves the rest of the circuit reachable only
    through that one bottleneck. Sampling the pose spreads the start over the
    whole circuit and over speeds the car actually races at.

    Fields:
        * randomized: Whether reset samples a pose instead of using the canonical start.
        * lateral_fraction: Bound on the start offset as a fraction of half the track width.
        * heading_error: Bound on the start heading error, in radians.
        * speed_fraction: Bound on the start speed as a fraction of the maximum speed.
    """

    randomized: bool = True
    lateral_fraction: float = 0.5
    heading_error: float = 0.2
    speed_fraction: float = 0.15


@dataclass(frozen=True, slots=True)
class CarConfig:
    """
    Physical configuration of the racing car.

    The car is a point mass on a kinematic bicycle, but three limits stop it
    from exploiting that abstraction. Tyre forces share one friction budget, so
    demanding longitudinal acceleration leaves less grip for cornering and the
    car understeers when the driver asks for more. Steering is rate limited, so
    the front wheels cannot flip between locks within one agent step.
    Aerodynamic drag, not the speed clamp, sets the achievable top speed.

    Acceleration is engine limited and therefore well below the friction budget,
    while braking is limited by the tyres and equals it: braking at the limit
    leaves nothing for cornering, which is exactly the trade the driver faces.

    Fields:
        * wheelbase: The distance between the front and rear axles, in meters.
        * max_acceleration: The engine-limited forward acceleration, in meters per second squared.
        * max_braking: The grip-limited braking deceleration, in meters per second squared.
        * max_steering_angle: The maximum steering angle, in degrees.
        * max_steering_rate: The maximum steering angle change, in degrees per second.
        * max_lateral_acceleration: The tyre friction budget, in meters per second squared.
        * max_speed: The speed at which drag cancels full throttle, in meters per second.
    """

    wheelbase: float = 3.6
    max_acceleration: float = 9.26
    max_braking: float = 20.0
    max_steering_angle: float = 30.0
    max_steering_rate: float = 180.0
    max_lateral_acceleration: float = 20.0
    max_speed: float = 70.0

    @property
    def drag_coefficient(self) -> float:
        """
        Return the quadratic drag that makes `max_speed` the terminal speed.

        Deriving drag from the two limits already declared keeps the top speed
        interpretable and avoids introducing an unmotivated constant.
        """
        return self.max_acceleration / self.max_speed**2


@dataclass(frozen=True, slots=True)
class TrackGenerationConfig:
    """
    Configuration for procedural track generation and geometric validation.
    The track is generated starting from a circle of radius ``base_radius``
    and then jittered radially and angularly to create a smooth, closed track.

    Fields:
        * n_checkpoints: The number of checkpoints used to define the track's shape. Must be at least 3.
        * base_radius: The base radius of the track, in meters.
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
    base_radius: float = 50.0
    radial_jitter: float = 0.25
    angular_jitter: float = 0.25
    sample_spacing: float = 0.5
    width: float = 12.0
    max_attempts: int = 100
    min_length: float = 200.0
    max_length: float = 600.0
    nonlocal_centerline_margin: float = 2.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Configuration of the reward function.

    The coefficients are chosen so that the dense progress term dominates the
    return. A policy that drives further before crashing must always score
    higher than one that crashes earlier, both must score higher than one that
    never leaves the start line, and a fast lap must score clearly higher than a
    slow one. This requires three orderings:

    * ``crash_penalty`` must stay below the cost of idling to the time limit
      (``time_penalty_rate * agent_timestep * max_episode_steps``), otherwise
      standing still strictly dominates every attempt to drive.
    * ``progress_coefficient`` must stay well above ``crash_penalty``, otherwise
      reaching further along the track is invisible next to the terminal event.
    * ``lap_time_bonus`` must be a visible share of a completed lap's return,
      otherwise the task rewards finishing rather than racing, and a policy that
      crawls around the circuit scores almost as well as one that attacks it.

    The per-step time penalty alone cannot carry the third ordering: raising it
    far enough to matter would eventually make an early crash cheaper than a
    slow lap, re-inverting the first ordering. Paying the bonus only on
    completion keeps lap time expensive without ever making crashing attractive.

    Fields:
        * finish_reward: The reward given to the agent for successfully completing the track.
        * lap_time_bonus: The additional completion reward scaled by the unused episode clock.
        * crash_penalty: The penalty given to the agent for crashing into an obstacle or going off-track.
        * time_penalty_rate: The penalty rate applied to the agent's reward for each second of simulation time.
        * progress_coefficient: The coefficient applied to the agent's reward based on its progress along the track.
    """

    finish_reward: float = 100.0
    lap_time_bonus: float = 100.0
    crash_penalty: float = 5.0
    time_penalty_rate: float = 1.0
    progress_coefficient: float = 100.0


@dataclass(frozen=True, slots=True)
class FrenetObservationConfig:
    """
    Configuration of the Frenet observation.

    Fields:
        * lookahead_base: The base lookahead distance for the Frenet observation, in meters.
        * lookahead_speed_factor: The speed factor used to adjust the lookahead distance based on the speed, in seconds.
    """

    lookahead_base: float = 5.0
    lookahead_speed_factor: float = 0.7
