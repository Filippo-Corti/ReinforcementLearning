"""Typed and validated configuration for the racing environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Timing and episode-length settings."""

    agent_timestep_s: float = 0.04
    physics_timestep_s: float = 0.01
    physics_substeps: int = 4
    max_episode_steps: int = 5_000

    def __post_init__(self) -> None:
        if not isfinite(self.agent_timestep_s) or self.agent_timestep_s <= 0:
            raise ValueError("agent_timestep_s must be finite and positive.")
        if not isfinite(self.physics_timestep_s) or self.physics_timestep_s <= 0:
            raise ValueError("physics_timestep_s must be finite and positive.")
        if type(self.physics_substeps) is not int or self.physics_substeps <= 0:
            raise ValueError("physics_substeps must be a positive integer.")
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer.")

        integrated_timestep = self.physics_timestep_s * self.physics_substeps
        if not isclose(
            self.agent_timestep_s,
            integrated_timestep,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "agent_timestep_s must equal "
                "physics_timestep_s * physics_substeps."
            )


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    """Physical limits of the kinematic vehicle."""

    wheelbase_m: float = 3.6
    max_acceleration_m_per_s2: float = 9.26
    max_steering_angle_deg: float = 30.0
    max_speed_m_per_s: float = 70.0

    def __post_init__(self) -> None:
        if not isfinite(self.wheelbase_m) or self.wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be finite and positive.")
        if (
            not isfinite(self.max_acceleration_m_per_s2)
            or self.max_acceleration_m_per_s2 <= 0
        ):
            raise ValueError(
                "max_acceleration_m_per_s2 must be finite and positive."
            )
        if (
            not isfinite(self.max_steering_angle_deg)
            or not 0 < self.max_steering_angle_deg < 90
        ):
            raise ValueError(
                "max_steering_angle_deg must be finite and between 0 and 90."
            )
        if not isfinite(self.max_speed_m_per_s) or self.max_speed_m_per_s <= 0:
            raise ValueError("max_speed_m_per_s must be finite and positive.")


@dataclass(frozen=True, slots=True)
class TrackGenerationConfig:
    """Procedural track-generation and geometry-validation settings."""

    n_checkpoints: int = 12
    base_radius_m: float = 250.0
    radial_jitter_fraction: float = 0.25
    angular_jitter_sectors: float = 0.25
    sample_spacing_m: float = 0.5
    width_m: float = 12.0
    max_attempts: int = 100
    min_length_m: float = 1_000.0
    max_length_m: float = 3_000.0
    nonlocal_centerline_margin_m: float = 2.0

    def __post_init__(self) -> None:
        if type(self.n_checkpoints) is not int or self.n_checkpoints < 3:
            raise ValueError("n_checkpoints must be an integer of at least 3.")
        if not isfinite(self.base_radius_m) or self.base_radius_m <= 0:
            raise ValueError("base_radius_m must be finite and positive.")
        if (
            not isfinite(self.radial_jitter_fraction)
            or not 0 <= self.radial_jitter_fraction < 1
        ):
            raise ValueError(
                "radial_jitter_fraction must be finite and in [0, 1)."
            )
        if (
            not isfinite(self.angular_jitter_sectors)
            or not 0 <= self.angular_jitter_sectors < 0.5
        ):
            raise ValueError(
                "angular_jitter_sectors must be finite and in [0, 0.5)."
            )
        if not isfinite(self.sample_spacing_m) or self.sample_spacing_m <= 0:
            raise ValueError("sample_spacing_m must be finite and positive.")
        if not isfinite(self.width_m) or self.width_m <= 0:
            raise ValueError("width_m must be finite and positive.")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")
        if not isfinite(self.min_length_m) or self.min_length_m <= 0:
            raise ValueError("min_length_m must be finite and positive.")
        if not isfinite(self.max_length_m) or self.max_length_m <= 0:
            raise ValueError("max_length_m must be finite and positive.")
        if self.min_length_m >= self.max_length_m:
            raise ValueError("min_length_m must be less than max_length_m.")
        if (
            not isfinite(self.nonlocal_centerline_margin_m)
            or self.nonlocal_centerline_margin_m < 0
        ):
            raise ValueError(
                "nonlocal_centerline_margin_m must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Coefficients used by the environment reward."""

    finish_reward: float = 10.0
    crash_penalty: float = 20.0
    time_penalty_rate_per_s: float = 0.05
    progress_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.finish_reward) or self.finish_reward <= 0:
            raise ValueError("finish_reward must be finite and positive.")
        if not isfinite(self.crash_penalty) or self.crash_penalty <= 0:
            raise ValueError("crash_penalty must be finite and positive.")
        if (
            not isfinite(self.time_penalty_rate_per_s)
            or self.time_penalty_rate_per_s < 0
        ):
            raise ValueError(
                "time_penalty_rate_per_s must be finite and non-negative."
            )
        if (
            not isfinite(self.progress_coefficient)
            or self.progress_coefficient < 0
        ):
            raise ValueError(
                "progress_coefficient must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class FrenetObservationConfig:
    """Parameters of the velocity-dependent curvature preview."""

    lookahead_base_m: float = 5.0
    lookahead_speed_factor_s: float = 0.7

    def __post_init__(self) -> None:
        if not isfinite(self.lookahead_base_m) or self.lookahead_base_m <= 0:
            raise ValueError("lookahead_base_m must be finite and positive.")
        if (
            not isfinite(self.lookahead_speed_factor_s)
            or self.lookahead_speed_factor_s < 0
        ):
            raise ValueError(
                "lookahead_speed_factor_s must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Complete configuration of environment-owned behaviour."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    track: TrackGenerationConfig = field(default_factory=TrackGenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: FrenetObservationConfig = field(
        default_factory=FrenetObservationConfig
    )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic plain-dictionary representation."""
        return asdict(self)
