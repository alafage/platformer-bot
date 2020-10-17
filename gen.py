import torch

from entities.dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(512, 720, 6).to(device)
PATH = "./models/policy_net.pth"
torch.save(policy_net.state_dict(), PATH)
print(">>> network created")
