"""One-step continuous control fixture with a known improving action direction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PositiveThrottleEnvironment:
    """
    Reward the first bounded action component in a deterministic one-step task.

    The sole observation is constant and every action ends the episode. The
    optimal deterministic first action is positive, while the second action is
    irrelevant. This keeps REINFORCE validation focused on policy gradients.

    Fields:
        * observation: Constant network input returned after every reset and step.
    """

    def __init__(self) -> None:
        """
        Initialize the constant one-dimensional observation.
        """
        self.observation = np.asarray((1.0,), dtype=np.float32)

    def reset(self) -> NDArray[np.float32]:
        """
        Begin one deterministic one-step episode.
        """
        return self.observation.copy()

    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float]:
        """
        End the episode and return its first-action-component reward.
        """
        bounded_action = np.asarray(action, dtype=np.float32)
        return self.observation.copy(), float(bounded_action[0])
