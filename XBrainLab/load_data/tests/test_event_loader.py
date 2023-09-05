from XBrainLab.load_data import Raw, EventLoader

from .test_raw import _set_event, raw, mne_raw, epoch, mne_epoch

import pytest
import numpy as np

@pytest.fixture
def mock_txt(mocker):
    mock_generator = mocker.patch('builtins.open', mocker.mock_open(read_data='1 2 3 4'))
    return mock_generator

def test_event_loader(raw):
    _set_event(raw)
    event_loader = EventLoader(raw)
    
    with pytest.raises(ValueError, match='No label has been loaded.'):
        event_loader.create_event({})

    with pytest.raises(AssertionError):
        event_loader.apply()

def _create_event(event_loader, raw):
    with pytest.raises(AssertionError):
        event_loader.apply()
    with pytest.raises(ValueError, match='Event name cannot be empty.'):
        event_loader.create_event({0: ''})

    event_loader.create_event({1: 'new 1', 2: 'new 2', 3: 'new 3', 4: 'new 4'})
    event_loader.apply()

    events, event_id = raw.get_event_list()
    assert len(events) == 4
    assert len(event_id) == 4
    for i in range(4):
        assert event_id['new ' + str(i + 1)] == i + 1
        assert events[i, -1] == i + 1
        assert events[i, 0] == i

def test_load_txt(raw, mock_txt):
    event_loader = EventLoader(raw)
    event_loader.read_txt('tests/0.txt')
    _create_event(event_loader, raw)
    
@pytest.fixture
def mock_mat(mocker):
    def mock_mat_generator(return_value):
        mock = mocker.patch('scipy.io.loadmat', autospec=True)
        mock.return_value = return_value        
    return mock_mat_generator

def test_mat_1d(raw, mock_mat):
    event_loader = EventLoader(raw)

    mock_mat({
        'label': np.array([1, 2, 3, 4]),
    })
    event_loader.read_mat('tests/0.mat')
    _create_event(event_loader, raw)

def test_mat_3d(raw, mock_mat):
    event_loader = EventLoader(raw)
    mock_mat({
        'label': np.array([[0, 0, 1],
                           [1, 0, 2],
                           [2, 0, 3],
                           [3, 0, 4]]),
    })
    event_loader.read_mat('tests/0.mat')
    _create_event(event_loader, raw)

def test_mat_malformed(raw, mock_mat):
    event_loader = EventLoader(raw)
    mock_mat({
        'label': np.array([[[0]]])
    })
    with pytest.raises(ValueError, match='Either 1d or 2d array is expected.'):
        event_loader.read_mat('tests/0.mat')


def test_mat_multi_key(raw, mock_mat):
    event_loader = EventLoader(raw)
    mock_mat({
        'label': np.array([[0, 0, 1]]),
        'error': True
    })
    with pytest.raises(ValueError, match='Mat file should contain exactly one variable.'):
        event_loader.read_mat('tests/0.mat')

def test_create_event_inconsistent(epoch, mock_mat):
    event_loader = EventLoader(epoch)
    mock_mat({
        'label': np.array([[0, 0, 1]])
    })
    event_loader.read_mat('tests/0.mat')
    with pytest.raises(ValueError, match='Inconsistent number of events.*'):
        event_loader.create_event({0: 'new 1',
                                   1: 'new 2'})

