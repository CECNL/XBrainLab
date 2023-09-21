from XBrainLab import preprocessor
from XBrainLab.load_data import Raw

from .test_base import (
    raw, # noqa: F401
    _generate_mne, base_fs 
)

import mne
import pytest
import numpy as np


# epoch
@pytest.fixture
def mne_epoch():
    mne_raw = _generate_mne(base_fs, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')

    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    epochs = mne.Epochs(
        mne_raw, events, event_id, tmin=0, tmax=5, baseline=None, preload=True
    )
    return epochs

@pytest.fixture
def epoch(mne_epoch):
    path = 'tests/test_data/sub-01_ses-01_task-rest_eeg.fif'    
    return Raw(path, mne_epoch)

# edit event
@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_channel_selection(target, request):
    target = request.getfixturevalue(target)
    processor = preprocessor.ChannelSelection([target])

    with pytest.raises(ValueError):
        processor.data_preprocess([])

    processor.data_preprocess(['Fp1', 'Fp2'])
    result = processor.get_preprocessed_data_list()[0]
    assert target.get_nchan() == 4
    assert result.get_nchan() == 2
    assert result.get_preprocess_history()[0] == 'Select 2 Channel'

def test_edit_event_name_raw(raw): # noqa: F811
    with pytest.raises(
        ValueError, match="Event name can only be edited for epoched data"
    ):
        preprocessor.EditEventName([raw])

def test_edit_event_name_epoch(epoch):
    processor = preprocessor.EditEventName([epoch])
    with pytest.raises(
        AssertionError, match="New event name not found in old event name."
    ):
        processor.data_preprocess({'ff': 'a'})
    with pytest.raises(AssertionError, match="No Event name updated."):
        processor.data_preprocess({'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'})
    with pytest.raises(ValueError, match="Duplicate event name: d"):
        processor.data_preprocess({'a': 'a', 'b': 'b', 'c': 'd', 'd': 'd'})
    
    # all match
    processor.data_preprocess({'a': 'a', 'b': 'b', 'c': 'c', 'd': 'e'})
    result = processor.get_preprocessed_data_list()[0]
    assert epoch.get_event_name_list_str() == 'a,b,c,d'
    assert result.get_event_name_list_str() == 'a,b,c,e'
    assert result.get_preprocess_history()[0] == 'Update 1 event names'

    # miss at new event
    processor.data_preprocess({'a': 'a', 'b': 'b', 'e': 'f'})
    result = processor.get_preprocessed_data_list()[0]
    assert epoch.get_event_name_list_str() == 'a,b,c,d'
    assert result.get_event_name_list_str() == 'a,b,c,f'
    assert result.get_preprocess_history()[1] == 'Update 1 event names'

# export
@pytest.mark.parametrize('target_str', ['raw', 'epoch'])
def test_export(target_str, request, mocker):
    mocked_savemat = mocker.patch('scipy.io.savemat')
    target = request.getfixturevalue(target_str)
    # to ensure history is not empty
    processor2 = preprocessor.ChannelSelection([target]) 
    processor2.data_preprocess(['Fp1', 'Fp2'])

    processor = preprocessor.Export(processor2.get_preprocessed_data_list())
    processor.data_preprocess('tests/test_data')
    
    args, _ = mocked_savemat.call_args
    assert args[0] == 'tests/test_data/Sub-0_Sess-0.mat'
    assert 'x' in args[1]
    if target_str == 'epoch':
        assert 'y' in args[1]
    assert 'history' in args[1]

# filtering
@pytest.mark.parametrize('target_str', ['raw', 'epoch'])
def test_filtering(target_str, request):
    target = request.getfixturevalue(target_str)
    processor = preprocessor.Filtering([target])
    fs = 100
    processor.data_preprocess(1, fs)
    result = processor.get_preprocessed_data_list()[0]
    assert target.get_filter_range() == (0, base_fs / 2)
    assert result.get_filter_range() == (1, fs)
    assert result.get_preprocess_history()[0] == 'Filtering 1 ~ ' + str(fs)

# resample
@pytest.mark.parametrize('target_str', ['raw', 'epoch'])
def test_resample(target_str, request):
    target = request.getfixturevalue(target_str)
    processor = preprocessor.Resample([target])
    fs = 50
    processor.data_preprocess(fs)
    result = processor.get_preprocessed_data_list()[0]
    assert target.get_sfreq() == base_fs
    assert result.get_sfreq() == fs
    assert result.get_preprocess_history()[-1] == 'Resample to ' + str(fs)

    events = np.array([[15, 0, 1], [20, 0, 2], [30, 0, 3], [40, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    # add event
    target.set_event(events, event_id)
    
    processor = preprocessor.Resample([target])
    fs = 100
    processor.data_preprocess(fs)
    result = processor.get_preprocessed_data_list()[0]
    assert result.get_sfreq() == fs
    assert result.get_preprocess_history()[-1] == 'Resample to ' + str(fs)
    # onset doesn't matter for already epoched data
    if target_str == 'raw':
        new_events, _ = result.get_event_list()
        assert np.allclose(new_events[:, 0] * 5, events[:, 0])

# epoch
@pytest.mark.parametrize('method', [preprocessor.TimeEpoch, preprocessor.WindowEpoch])
def test_epoch_wrong_type(method, epoch):
    with pytest.raises(ValueError, match="Only raw data can be epoched, got epochs"):
        method([epoch])

@pytest.mark.parametrize('method', [preprocessor.TimeEpoch, preprocessor.WindowEpoch])
def test_epoch_wrong_events(method, raw): # noqa: F811
    with pytest.raises(ValueError, match="No event markers found.*"):
        method([raw])
        
def test_sliding_epoch_error(raw): # noqa: F811
    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    raw.set_event(events, event_id)
    with pytest.raises(ValueError, match="Should only contain single event label.*"):
        preprocessor.WindowEpoch([raw])
    
# time epoch
@pytest.fixture
def annotated_raw():
    info = mne.create_info(ch_names=['Fp1', 'Fp2'],
                           sfreq=1,
                           ch_types='eeg')
    data = np.array([[1, 3 ,5, 7, 9, 11, 13, 15, 17, 19],
                     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]])
    mne_raw = mne.io.RawArray(data, info)
    annotated_raw = Raw('tests/test_data/sub-01_ses-01_task-rest_eeg.fif', mne_raw)
    events = np.array([[1, 0, 1], [3, 0, 2], [5, 0, 3], [7, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    annotated_raw.set_event(events, event_id)
    return annotated_raw

def test_time_epoch_no_epochs(annotated_raw):    
    processor = preprocessor.TimeEpoch([annotated_raw])
    with pytest.raises(ValueError, match="No event markers found."):
        processor.data_preprocess(
            baseline=None, selected_event_names=['g'], tmin=0, tmax=1
        )
    

def test_time_epoch_without_baseline(annotated_raw):    
    processor = preprocessor.TimeEpoch([annotated_raw])
    processor.data_preprocess(
        baseline=None, selected_event_names=['a', 'b', 'c', 'd'], tmin=0, tmax=1
    )
    result = processor.get_preprocessed_data_list()[0]
    assert result.get_event_name_list_str() == 'a,b,c,d'
    assert result.get_mne().get_data().shape == (4, 2, 2)
    assert np.allclose(
        result.get_mne().get_data(), 
        np.array([[[3, 5], [4, 6]], [[7, 9], [8, 10]], 
                  [[11, 13], [12, 14]], [[15, 17], [16, 18]]])
    )
    assert (
        result.get_preprocess_history()[0] == 
        'Epoching 0 ~ 1 by event (None baseline)'
    )

def test_time_epoch_with_baseline(annotated_raw):
    processor = preprocessor.TimeEpoch([annotated_raw])
    processor.data_preprocess(
        baseline=(-1, 0), 
        selected_event_names=['a', 'b', 'c', 'd'], 
        tmin=0, tmax=1
    )
    result = processor.get_preprocessed_data_list()[0]
    assert result.get_event_name_list_str() == 'a,b,c,d'
    assert result.get_mne().get_data().shape == (4, 2, 2)
    assert np.allclose(
        result.get_mne().get_data(), 
        np.array([[[0, 2], [0, 2]], [[0, 2], [0, 2]], 
                  [[0, 2], [0, 2]], [[0, 2], [0, 2]]])
    )
    assert (
        result.get_preprocess_history()[0] == 
        'Epoching 0 ~ 1 by event ((-1, 0) baseline)'
    )

def test_window_epoch(annotated_raw):
    events = np.array([[0, 0, 1]])
    event_id = {'a': 1}
    annotated_raw.set_event(events, event_id)
    processor = preprocessor.WindowEpoch([annotated_raw])
    processor.data_preprocess(duration=2, overlap=1)
    result = processor.get_preprocessed_data_list()[0]
    assert result.get_event_name_list_str() == 'a'
    assert result.get_mne().get_data().shape == (9, 2, 2)
    assert np.allclose(
        result.get_mne().get_data(), 
        np.array([[[1, 3], [2, 4]], [[3, 5], [4, 6]], 
                  [[5, 7], [6, 8]], [[7, 9], [8, 10]], 
                  [[9, 11], [10, 12]], [[11, 13], [12, 14]], 
                  [[13, 15], [14, 16]], [[15, 17], [16, 18]], 
                  [[17, 19], [18, 20]]])
    )
    assert (
        result.get_preprocess_history()[0] == 
        'Epoching 2s (1s overlap) by sliding window'
    )

@pytest.mark.xfail
def test_normalization_zero_min():
    raise NotImplementedError

@pytest.mark.xfail
def test_normalization_zero_min_minmax():
    raise NotImplementedError