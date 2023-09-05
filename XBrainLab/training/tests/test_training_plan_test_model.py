from XBrainLab.training.training_plan import to_holder, _test_model, _eval_model, EvalRecord

from torch.utils.data import DataLoader

import torch
import pytest
import numpy as np

@pytest.mark.parametrize('shuffle', [True, False])
def test_to_holder(shuffle):
    device = 'cpu'
    length = 3000
    X = np.arange(length).reshape(-1, 1)
    y = np.arange(length)

    bs = 128
    dataloader = to_holder(X, y, device, bs, shuffle)

    # Perform assertions
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == bs

    sample_x, sample_y = list(dataloader)[0]
    assert sample_x.dtype == torch.float32
    assert sample_y.dtype == torch.int64
    sequence = torch.arange(bs, dtype=torch.float32).reshape(-1, 1)
    if shuffle:
        with pytest.raises(AssertionError):
            torch.testing.assert_close(sample_x, sequence)
    else:
        torch.testing.assert_close(sample_x, sequence)

def test_to_holder_empty():
    X = np.array([])
    y = np.array([])
    device = 'cpu'
    bs = 128
    shuffle = True
    assert to_holder(X, y, device, bs, shuffle) == None

CLASS_NUM = 4
ERROR_NUM = 3
SAMPLE_NUM = CLASS_NUM
REPEAT = 5
TOTAL_NUM = SAMPLE_NUM * REPEAT
BS = 2

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(CLASS_NUM, CLASS_NUM)
        self.fc.weight.data = torch.diag(torch.ones(CLASS_NUM))
        self.fc.bias.data = torch.zeros_like(self.fc.bias.data)

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x

@pytest.fixture
def full_y():
    return np.arange(SAMPLE_NUM).repeat(REPEAT)

@pytest.fixture
def y(full_y):
    y = full_y.copy()
    y[:ERROR_NUM] += 1
    y[:ERROR_NUM] %= CLASS_NUM
    return y

@pytest.fixture
def dataloader(full_y, y):
    """
    X = [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
          ...
         [0, 0, 0, 1]]
    ground truth = [0, 0, 0, 0, 0, 1, ...]
    y = [1, 1, 1, 0, 0, 1, ...]
    """
    X = np.zeros((TOTAL_NUM, CLASS_NUM))
    for idx, gt in enumerate(full_y):
        X[idx, gt] = 1
    
    device = 'cpu'
    shuffle = False
    return to_holder(X, y, device, BS, shuffle)

@pytest.fixture
def loss_avg():
    criterion = torch.nn.CrossEntropyLoss()
    error_loss = criterion(torch.Tensor([[0, 0, 0, 1]]), torch.Tensor([0]).long()).item()
    correct_loss = criterion(torch.Tensor([[1, 0, 0, 0]]), torch.Tensor([0]).long()).item()
    loss = np.ones((TOTAL_NUM)) * correct_loss
    loss[:ERROR_NUM] = error_loss
    loss_avg = []
    for i in range(0, TOTAL_NUM, BS):
        loss_avg.append(loss[i:i+BS].mean())
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def test_test_model(dataloader, loss_avg):
    model = FakeModel()
    criterion = torch.nn.CrossEntropyLoss()
    test_dict = _test_model(model, dataloader, criterion)

    assert test_dict.keys() == {'loss', 'accuracy', 'auc'}
    assert test_dict['accuracy'] == (TOTAL_NUM - ERROR_NUM) / (TOTAL_NUM) * 100
    assert np.isclose(test_dict['loss'], loss_avg)

def test_test_model_auc(mocker, dataloader):
    raise NotImplementedError


def test_eval_model(mocker, dataloader, y, full_y):
    model = FakeModel()
    model.eval()
    eval_record_mock = mocker.patch('XBrainLab.training.record.eval.EvalRecord.__init__',
                                    return_value=None)
    eval_model_mock = mocker.patch.object(model, 'eval')
    result = _eval_model(model, dataloader)
    eval_model_mock.assert_called_once()
    
    assert isinstance(result, EvalRecord)
    called_y, called_output, called_gradient = eval_record_mock.call_args[0]
    assert np.array_equal(called_y, y)
    assert np.array_equal(called_output.argmax(axis=-1), full_y)
    assert len(called_gradient) == CLASS_NUM
    
    expected_list = [
        (REPEAT - ERROR_NUM, CLASS_NUM),
        (REPEAT + ERROR_NUM, CLASS_NUM),
        (REPEAT, CLASS_NUM),
        (REPEAT, CLASS_NUM)
    ]
    for g, expected_shape in zip(called_gradient, expected_list):
        assert called_gradient[g].shape == expected_shape