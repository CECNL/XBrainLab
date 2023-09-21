from __future__ import annotations

import random

import numpy as np
import torch
from numpy.random import MT19937, RandomState, SeedSequence


def set_seed(seed: int | None = None) -> int:
    """Set seed for reproducibility and return the seed value."""
    if seed is None:
        seed = torch.seed()

    # random
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # numpy
    rs = RandomState(MT19937(SeedSequence(seed)))
    np.random.set_state(rs.get_state())

    return seed

def get_random_state() -> tuple:
    """Returns the random state of torch, random and numpy"""
    return torch.get_rng_state(), random.getstate(), np.random.get_state()

def set_random_state(state: tuple) -> None:
    """Sets the random state of torch, random and numpy"""
    torch_state, random_state, np_state = state

    torch.torch.set_rng_state(torch_state)
    random.setstate(random_state)
    np.random.set_state(np_state)
