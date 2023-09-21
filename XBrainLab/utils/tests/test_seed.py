import numpy as np
import torch

from XBrainLab.utils import seed


def test_set_seed():
    result = seed.set_seed()
    seed_target = 42
    assert isinstance(result, int)
    assert seed.set_seed(seed_target) == seed_target

def test_get_random_state():
    result = seed.get_random_state()
    tuple_length = 3
    assert isinstance(result, tuple)
    assert len(result) == tuple_length
    assert isinstance(result[0], torch.ByteTensor)
    assert isinstance(result[1], tuple)
    assert isinstance(result[2], tuple)

def test_set_random_state():
    state = seed.get_random_state()
    seed.set_random_state(state)
    result = seed.get_random_state()
    # torch
    np.allclose(state[0], result[0])
    # random
    assert state[1] == result[1]
    # numpy
    for s, r in zip(state[2], result[2]):
        if isinstance(s, np.ndarray):
            assert (s == r).all()
        else:
            assert s == r
