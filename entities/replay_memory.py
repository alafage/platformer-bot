class ReplayMemory:
    def __init__(self, capacity):

        self.capacity = capacity
        self.memory = []
