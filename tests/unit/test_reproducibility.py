import random

import numpy as np
import torch

from src.training.reproducibility import set_seed


def test_set_seed_repeats_python_numpy_and_torch_values():
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
