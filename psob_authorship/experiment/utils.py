import torch
import numpy as np


def make_experiment_reproducible(random_seed: int):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
