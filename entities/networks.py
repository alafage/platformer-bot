from collections import OrderedDict

import torch.nn as nn


class PlatformerNet(nn.Module):
    """ TODO
    """

    def __init__(self, height, width, outputs):
        super(PlatformerNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    # 3x512x720
                    ("conv1", nn.Conv2d(3, 64, 8, 8)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    # 64x32x45
                    ("conv2", nn.Conv2d(64, 128, 5, 1, padding=2)),
                    ("bnor2", nn.BatchNorm2d(128)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # 128x32x45
                    ("conv3", nn.Conv2d(128, 256, 3, 1, padding=1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("pool3", nn.MaxPool2d(2, 2)),
                    # 256x16x22
                    ("conv4", nn.Conv2d(256, 512, 3, 1, padding=1)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("pool4", nn.MaxPool2d(2, 2)),
                    # 512x8x11
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(512 * 8 * 11, 4096)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(4096, 2048)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("fc3", nn.Linear(2048, 1024)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("fc4", nn.Linear(1024, outputs)),
                ]
            )
        )

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.size()
        x = x.view(-1, c * h * w)
        x = self.classifier(x)
        return x
