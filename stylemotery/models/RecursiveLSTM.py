import torch

from torch import nn


class RecursiveLSTM(nn.Module):
    def __init__(self, configuration):
        super(RecursiveLSTM, self).__init__()
        self.configuration = configuration

    def forward(self, input):
        output = torch.zeros(len(self.configuration.classes))
        output[0] = 1
        return output
