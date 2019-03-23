import torch

from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=9, output=40, nonlin=torch.tanh):
        super(Model, self).__init__()

        self.nonlin = nonlin
        self.input_hidden = nn.Linear(input_dim, 150)
        self.hidden_output = nn.Linear(150, output)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.input_hidden(x))
        x = self.nonlin(self.hidden_output(x))
        x = F.softmax(x, dim=-1)
        return x
