from XBrainLab import model_base

import torch
import pytest

@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("channel", [1, 2, 22, 23])
@pytest.mark.parametrize("samples", [1000, 1001, 1024])
@pytest.mark.parametrize("sfreq", [128, 256, 501])
@pytest.mark.parametrize("model_class_str", [
    m 
    for m in dir(model_base) 
    if not m.startswith("_") and isinstance(getattr(model_base, m), type)]
)
def test_model_base(n_classes, channel, samples, sfreq, model_class_str):
    model_class = getattr(model_base, model_class_str)
    model = model_class(n_classes, channel, samples, sfreq)
    input = torch.randn(1, channel, samples)
    model(input)
