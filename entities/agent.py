import random

import numpy as np
import torch


class Agent:
    def __init__(self, n_actions):
        """ TODO
        """
        self.n_actions = n_actions

    def observe_env(self, env, resize, device):
        """ TODO
        """
        # Gets
        screen = env.render(mode="rgb_array").transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        # this doesn't require a copy
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(device)

    def select_action(
        self, policy, state, exploration_rate, device=torch.device("cpu")
    ):
        """ TODO
        """
        sample = random.random()
        if sample >= exploration_rate:
            with torch.no_grad():
                return policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device,
                dtype=torch.long,
            )
