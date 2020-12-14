from typing import Any, TypeVar

import torch

TransitionTypeVar = TypeVar("TransitionTypeVar", bound="Transition")


class Transition(object):
    """ Transition object
    """

    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        reward: float,
    ) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Transition):
            equal_states = torch.equal(self.state, other.state)
            equal_actions = self.action == other.action
            equal_next_states = torch.equal(self.next_state, other.next_state)
            equal_rewards = self.reward == other.reward

            return (
                equal_states
                and equal_actions
                and equal_next_states
                and equal_rewards
            )
        else:
            return False

    def compress(self) -> TransitionTypeVar:
        """ Returns Transition with compressed states matrices.
        """
        ...
