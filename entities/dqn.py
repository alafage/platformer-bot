import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, height, width, outputs):
        super(DQN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=8, stride=8
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            """ TODO change fct name
            """
            return (size - (kernel_size - 1) - 1) // stride + 1

        # width of the picture after the convolutions
        convw = conv2d_size_out(conv2d_size_out(width, 8, 8), 2, 2)
        # height of the picture after the convolutions
        convh = conv2d_size_out(conv2d_size_out(height, 8, 8), 2, 2)
        linear_input_size = convw * convh * 4
        self.lin1 = nn.Linear(linear_input_size, linear_input_size // 2)
        self.lin2 = nn.Linear(linear_input_size // 2, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        x = self.lin2(x)
        return x
