from XBrainLab.load_data import Raw, RawDataLoader
from XBrainLab import XBrainLab

from .test_raw import _generate_mne, _set_event

import mne
import pytest
import numpy as np

def test_raw_data_loader():
    raw = Raw('tests/0.fif', _generate_mne(500, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg'))
    assert len(RawDataLoader()) == 0
    # no event
    with pytest.raises(ValueError):
        RawDataLoader([raw])
    # with event
    _set_event(raw)
    assert len(RawDataLoader([raw])) == 1
    # check empty list creation
    assert len(RawDataLoader()) == 0

def test_raw_data_loader_validate():
    with pytest.raises(ValueError):
        assert RawDataLoader().validate()

def _generate_epoch(name, raw_mne, duration):
    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    return Raw(
        f'tests/{name}.fif',
        mne.Epochs(
            raw_mne, events, event_id, 
            tmin=0, tmax=duration, baseline=None, preload=True
        )
    )
    
def test_raw_data_loader_append():
    raw_mne = _generate_mne(500, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')
    raw_1 = _generate_epoch('1', raw_mne, 0.1)
    raw_2 = _generate_epoch(
        '2', _generate_mne(500, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg'), 0.1
    )

    _set_event(raw_1)
    _set_event(raw_2)

    raw_data_loader = RawDataLoader()

    raw_data_loader.append(raw_1)
    assert len(raw_data_loader) == 1
    raw_data_loader.append(raw_2)
    assert len(raw_data_loader) == 2

    assert raw_data_loader.get_loaded_raw("empty") is None
    assert raw_data_loader.get_loaded_raw("tests/1.fif") == raw_1
    assert raw_data_loader.get_loaded_raw("tests/2.fif") == raw_2


def test_raw_data_loader_append_error():
    raw_mne = _generate_mne(500, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')
    raw_1 = _generate_epoch('1', raw_mne, 0.1)
    
    _set_event(raw_1)

    raw_miss_channel = _generate_epoch(
        'mc', 
        _generate_mne(500, ['Fp1', 'Fp2', 'F3'], 'eeg'), 0.1
    )
    raw_miss_sf = _generate_epoch(
        'ms', 
        _generate_mne(5, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg'), 0.1
    )
    raw_miss_duration = _generate_epoch('ms', raw_mne, 0.2)
    raw_miss_type = Raw('test/mt.fif', raw_mne)
  
    raw_data_loader = RawDataLoader()
    raw_data_loader.append(raw_1)

    assert len(raw_data_loader) == 1

    with pytest.raises(ValueError, match=r".*channel numbers inconsistent.*"):
        raw_data_loader.append(raw_miss_channel)
    
    with pytest.raises(ValueError, match=r".*sample frequency inconsistent.*"):
        raw_data_loader.append(raw_miss_sf)

    with pytest.raises(ValueError, match=r".*type inconsistent.*"):
        raw_data_loader.append(raw_miss_type)
    
    with pytest.raises(ValueError, match=r".*duration inconsistent.*"):
        raw_data_loader.append(raw_miss_duration)

def test_apply():
    raw_mne = _generate_mne(500, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')
    raw = Raw('test/mt.fif', raw_mne)
    _set_event(raw)
    
    lab = XBrainLab()
    RawDataLoader([raw]).apply(lab)