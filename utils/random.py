import random

import numpy as np
import torch


def set_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()  # can be used in pytorch dataloaders for reproducible sample selection when shuffle=True
    g.manual_seed(seed)

    return g
