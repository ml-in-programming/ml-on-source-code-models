import torch

from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim=19, output_dim=40, hidden_dim=150, nonlin=torch.tanh):
        super(Model, self).__init__()

        self.nonlin = nonlin
        self.input_hidden = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hidden_output = nn.Linear(hidden_dim, output_dim, bias=True)
        self.dimensions = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    def forward(self, x, **kwargs):
        x = self.nonlin(self.input_hidden(x))
        x = self.hidden_output(x)
        return x
