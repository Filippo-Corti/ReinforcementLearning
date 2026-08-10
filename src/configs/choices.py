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
