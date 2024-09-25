import torch
from torch import nn, Tensor


class MultiLayeredPerceptron(nn.Module):
    def __init__(self, dropout_probability: float = 0.2) -> None:
        super(MultiLayeredPerceptron, self).__init__()
        self.linear_0 = nn.Linear(in_features=784, out_features=512)
        self.dropout_0 = nn.Dropout(p=dropout_probability)
        self.relu_0 = nn.ReLU(inplace=True)

        self.linear_1 = nn.Linear(in_features=512, out_features=512)
        self.dropout_1 = nn.Dropout(p=dropout_probability)
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(in_features=512, out_features=512)
        self.dropout_2 = nn.Dropout(p=dropout_probability)
        self.relu_2 = nn.ReLU(inplace=True)

        self.output = nn.Linear(in_features=512, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_0(x)
        x = self.dropout_0(x)
        x = self.relu_0(x)

        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)

        x = self.linear_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)

        x = self.output(x)

        return x


class ConvolutionalNetwork(nn.Module):
    def __init__(self, dropout_probability: float = 0.2) -> None:
        super(ConvolutionalNetwork, self).__init__()
        self.conv2d_0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            bias=True,
            padding_mode='zeros'
        )
        self.dropout_0 = nn.Dropout(p=dropout_probability)
        self.relu_0 = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            bias=True,
            padding_mode='zeros'
        )
        self.dropout_1 = nn.Dropout(p=dropout_probability)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            dilation=(1, 1),
            bias=True,
            padding_mode='zeros'
        )
        self.dropout_2 = nn.Dropout(p=dropout_probability)
        self.relu_2 = nn.ReLU(inplace=True)

        self.output = nn.Linear(in_features=784, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d_0(x)
        x = self.dropout_0(x)
        x = self.relu_0(x)

        x = self.conv2d_1(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)

        x = x.view(size=(-1, 784))

        x = self.output(x)

        return x
