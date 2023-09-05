from XBrainLab.visualization.base import Visualizer
from XBrainLab.training.record import EvalRecord

import pytest
import numpy as np

def test_Visualizer():
    label = np.ones(10)
    output = np.ones((10, 2))
    gradient = {
        0: np.zeros((10, 2, 3, 4)),
        1: np.ones((10, 2, 3, 4)),
    }
    eval_record = EvalRecord(label, output, gradient)
    visualizer = Visualizer(eval_record, None)
    with pytest.raises(NotImplementedError):
        visualizer.get_plt()
    
    assert np.array_equal(visualizer.get_gradient(0), np.zeros((10, 2, 3, 4)))
    assert np.array_equal(visualizer.get_gradient(1), np.ones((10, 2, 3, 4)))