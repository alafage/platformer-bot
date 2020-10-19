from itertools import count

import cv2
import torch
import torchvision.transforms as T
from gymplatformer import make

import pygame
from entities.agent import Agent
from entities.dqn import DQN

PATH = "./models/policy_net.pth"
RECORDING = False
AVI_FILE = "./rec/level0.avi"
NETWORK = DQN
NB_ITER = 4

continuous_env = make("PlatformerEnv", ep_duration=2)

clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

continuous_env.reset()

resize = T.Compose(
    [
        T.ToPILImage(),
        # T.Resize((27,126)),
        T.ToTensor(),
    ]
)

# Get number of actions from gym action space
n_actions = continuous_env.action_space.n
# Initializes agent
agent = Agent(n_actions)

# init_screen = get_screen(discrete_env)
init_screen = agent.observe_env(continuous_env, resize, device)

# plt.figure()
# plt.imshow(init_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

_, _, screen_height, screen_width = init_screen.shape

# policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net = NETWORK(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load(PATH))
policy_net = policy_net.eval()

# for param in policy_net.parameters():
#     print(param.data)

if RECORDING:

    with torch.no_grad():
        img_array = []
        for i in range(NB_ITER):

            continuous_env.reset()
            # print('Init: ',discrete_env.state)
            continuous_env.render()
            state = agent.observe_env(continuous_env, resize, device)

            for t in count():
                # Sets number of fps
                clock.tick(20)

                img_array.append(continuous_env.render(mode="rgb_array"))

                # Select and perform an action
                action = agent.select_action(policy_net, state, float(0))

                states, reward, done = continuous_env.step(action.item())

                # displays environment
                continuous_env.render()

                next_state = agent.observe_env(continuous_env, resize, device)

                # Move to the next state
                state = next_state

                # break is episode completed
                if done:
                    break

        out = cv2.VideoWriter(
            AVI_FILE,
            cv2.VideoWriter_fourcc(*"XVID"),
            20,
            img_array[0].shape[:-1][::-1],
        )
        for i in range(len(img_array)):
            frame = cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()

else:

    with torch.no_grad():

        for i in range(NB_ITER):

            continuous_env.reset()
            # print('Init: ',discrete_env.state)
            continuous_env.render()
            state = agent.observe_env(continuous_env, resize, device)

            for t in count():
                # Sets number of fps
                clock.tick(20)

                # Select and perform an action
                action = agent.select_action(policy_net, state, float(0))
                states, reward, done = continuous_env.step(action.item())

                # displays environment
                continuous_env.render()

                next_state = agent.observe_env(continuous_env, resize, device)

                # Move to the next state
                state = next_state

                # break is episode completed
                if done:
                    break
