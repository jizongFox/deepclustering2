import torch.nn as nn

from .._prototype import _AbstractNetwork


class VATNetwork(_AbstractNetwork):
    def __init__(self, input_dim=3, num_classes=10, top_bn=True):
        super(VATNetwork, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        self.top_bn = top_bn
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.linear = nn.Linear(128, output_k)
        self.bn = nn.BatchNorm1d(output_k)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        if self.top_bn:
            output = self.bn(output)
        return output

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def num_classes(self):
        return self._num_classes
