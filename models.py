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
