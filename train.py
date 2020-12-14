# -*- coding: utf-8 -*-

import math
from itertools import count
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from entities.agent import Agent
from entities.dqn import DQN
from entities.replay_memory import ReplayMemory

PATH_IN = "./mdl/policy_net.pth"
PATH_OUT = "./mdl/policy_net_lvl4_bis.pth"
DATA_FILE = "./dat/scores_lvl4_bis.csv"
NETWORK = DQN

env = gym.make("gym_platformer:platformer-v0")

clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# INPUT EXTRACTION

resize = T.Compose(
    [
        T.ToPILImage(),
        # T.Resize((27,126)),
        T.ToTensor(),
    ]
)

# TRAINING

# Hyperparameters and utilities

BATCH_SIZE = 60
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
NUM_EP = 300

# Get screen size so that we can initialize layers correctly based on shape
# returned from GymEnv.
env.reset()
# Get number of actions from gym action space
n_actions = env.action_space.n
agent = Agent(n_actions)

init_screen = agent.observe_env(env, resize, device)
_, _, screen_height, screen_width = init_screen.shape


policy_net = NETWORK(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load(PATH_IN))
target_net = NETWORK(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_scores: List[float] = []


def plot_scores():
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(scores_t) >= 100:
    #     means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy)

    plt.pause(0.001)


# Training loop


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = memory.transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for i_episode in range(NUM_EP):
    print("ep: %d" % (i_episode), end="\r")
    # Initialize the environment and state
    env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    state = agent.observe_env(
        env, resize, device
    )  # current_screen - last_screen
    rewards = 0
    for t in count():
        # sets number of fps
        # clock.tick(20)
        # Select and perform an action
        exploration_rate = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1 * steps_done / EPS_DECAY
        )
        action = agent.select_action(policy_net, state, exploration_rate)
        steps_done += 1

        _, reward, done, _ = env.step(action.item(), t)
        rewards += reward
        reward = torch.tensor([reward], device=device)
        # Dislay view
        env.render()
        # Observe new state
        # if not done:
        #     next_state = get_screen() # current_screen - last_screen
        # else:
        #     next_state = None
        next_state = agent.observe_env(env, resize, device)
        # Store the transition in memory
        test = memory.push(state, action, next_state, reward)
        # print(test)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_scores.append(rewards)
            # plot_scores()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Complete")

for param in policy_net.parameters():
    print(param.data)

torch.save(policy_net.state_dict(), PATH_OUT)
print(">>> network saved")

scores = np.asarray(episode_scores)
np.savetxt(DATA_FILE, scores, delimiter=",")
print(">>> scores saved")

env.close()
