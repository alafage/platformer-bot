import random

from transition import Transition


class ReplayMemory(object):
    """ Replay Memory object
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, new_transition: Transition):
        """Saves a transition."""
        # TODO: compresses state matrices
        # FIXME: is that relevant
        for mem in self.memory:
            if mem:
                if new_transition == mem:
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
