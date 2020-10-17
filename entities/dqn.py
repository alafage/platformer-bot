import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, height, width, outputs):
        super(DQN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=9, stride=9
        )
        self.bn = nn.BatchNorm2d(4)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=9, stride=9):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # width of the picture after the convolutions
        convw = conv2d_size_out(width)
        # height of the picture after the convolutions
        convh = conv2d_size_out(height)
        linear_input_size = convw * convh * 4
        self.lin1 = nn.Linear(linear_input_size, convw * convh * 2)
        self.lin2 = nn.Linear(convw * convh * 2, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        x = self.lin2(x)
        return x
