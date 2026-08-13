"""Named choices shared by experiment configurations."""

from enum import StrEnum


class Algorithm(StrEnum):
    """
    Identify one project-owned policy-gradient algorithm.
    """

    REINFORCE = "reinforce"
    A2C = "a2c"
    PPO = "ppo"


class ObservationRepresentation(StrEnum):
    """
    Identify the observation representation supplied to an agent.
    """

    FRENET = "frenet"
    LIDAR = "lidar"


class RenderStyle(StrEnum):
    """
    Identify how the environment is drawn, which never affects what it does.

    The minimal style answers "where did the car go"; the broadcast style
    answers "what was it like to drive". Neither is read by the simulation.
    """

    MINIMAL = "minimal"
    BROADCAST = "broadcast"
