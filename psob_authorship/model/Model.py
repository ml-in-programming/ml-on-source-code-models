import torch

from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    HIDDEN_DIM = 150

    def __init__(self, input_dim=18, output=40, nonlin=torch.tanh):
        super(Model, self).__init__()

        self.nonlin = nonlin
        self.input_hidden = nn.Linear(input_dim, Model.HIDDEN_DIM, bias=True)
        self.hidden_output = nn.Linear(Model.HIDDEN_DIM, output, bias=True)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.input_hidden(x))
        x = self.nonlin(self.hidden_output(x))
        x = F.softmax(x, dim=-1)
        return x
