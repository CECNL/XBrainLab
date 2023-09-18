import torch
import numpy as np
from XBrainLab.utils import seed

def test_set_seed():
    result = seed.set_seed()
    assert isinstance(result, int)
    assert seed.set_seed(42) == 42

def test_get_random_state():
    result = seed.get_random_state()
    assert isinstance(result, tuple)
    assert len(result) == 3
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