import random
from collections import namedtuple

import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def equal(self, transition_a, transition_b):
        equalStates = torch.equal(transition_a.state, transition_b.state)
        equalActions = torch.equal(transition_a.action, transition_b.action)
        equalNewStates = torch.equal(
            transition_a.next_state, transition_b.next_state
        )
        equalRewards = torch.equal(transition_a.reward, transition_b.reward)

        return equalStates and equalActions and equalNewStates and equalRewards

    def push(self, *args):
        """Saves a transition."""
        new_transition = Transition(*args)
        for mem in self.memory:
            if mem:
                if self.equal(new_transition, mem):
                    return (False, len(self.memory))

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = new_transition
        self.position = (self.position + 1) % self.capacity
        return (True, len(self.memory))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
