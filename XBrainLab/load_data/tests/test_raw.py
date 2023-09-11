from XBrainLab.load_data import Raw

import pytest
import traceback
import mne
import numpy as np

base_fs = 50
base_duration = 1

def _generate_mne(fs, ch_names, ch_types, length = base_duration):
    info = mne.create_info(ch_names=ch_names,
                           sfreq=fs,
                           ch_types=ch_types)
    data = np.random.RandomState(0).randn(len(ch_names), fs * length)
    return mne.io.RawArray(data, info)

# raw without event
@pytest.fixture
def mne_raw():
    return _generate_mne(base_fs, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')

@pytest.fixture
def mne_raw_2():
    mne_raw = _generate_mne(50, ['O1', 'O2'], 'eeg')
    return mne_raw

@pytest.fixture
def raw(mne_raw):
    path = 'tests/test_data/sub-01_ses-01_task-rest_eeg.fif'    
    return Raw(path, mne_raw)

# common
def test_add_preprocess(raw):
    raw.add_preprocess('test')
    assert raw.get_preprocess_history() == ['test']

def test_parse_filename(raw, mocker):
    raw.parse_filename('sub-(?P<subject>[^_]*)_ses-(?P<session>[^_]*)_.*')
    assert raw.get_subject_name() == '01'
    assert raw.get_session_name() == '01'

    mock_print_exc = mocker.patch.object(traceback, "print_exc")
    raw.parse_filename('sub-(?P<subject>[^_]*_ses-(?P<session>[^_]*)_.*')
    mock_print_exc.assert_called_once()    

# set event
def _set_event(raw):
    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    # add event
    raw.set_event(events, event_id)

def test_set_event(raw):
    _set_event(raw)
    assert raw.has_event_str() == 'yes'
    assert raw.get_event_name_list_str() == 'a,b,c,d'
    # check event
    events, event_id = raw.get_event_list()
    assert len(events) == 4
    assert len(event_id) == 4

def test_set_event_error(raw):
    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    with pytest.raises(AssertionError):
        raw.set_event(np.array([1, 0, 1]), event_id)

    with pytest.raises(AssertionError):
        raw.set_event(np.array([[1, 0]]), event_id)
    
    with pytest.raises(TypeError):
        raw.set_event(events, [1, 2, 3])

def test_set_event_consistency(epoch):
    events = np.array([[1, 0, 1], [2, 0, 2]])
    event_id = {'a': 1, 'b': 2}
    with pytest.raises(AssertionError):
        epoch.set_event(events, event_id)

# raw
def test_mne_raw_info(mne_raw, raw):
    assert mne_raw == raw.get_mne()
    assert raw.get_tmin() == 0.0
    assert raw.get_nchan() == 4
    assert raw.get_sfreq() == base_fs
    assert raw.get_filter_range() == (0, base_fs / 2)
    assert raw.get_epochs_length() == 1
    assert raw.get_epoch_duration() == base_fs * base_duration
    assert raw.is_raw() == True

    assert raw.get_row_info() == ('sub-01_ses-01_task-rest_eeg.fif', '0', '0', 4, base_fs, 1, 'no')

def test_mne_raw_2_info(mne_raw_2, raw):
    raw.set_mne(mne_raw_2)
    assert mne_raw_2 == raw.get_mne()
    assert raw.get_tmin() == 0.0
    assert raw.get_nchan() == 2
    assert raw.get_sfreq() == 50
    assert raw.get_filter_range() == (0, 25)
    assert raw.get_epochs_length() == 1
    assert raw.get_epoch_duration() == 50 * base_duration
    assert raw.is_raw() == True

    assert raw.get_row_info() == ('sub-01_ses-01_task-rest_eeg.fif', '0', '0', 2, 50, 1, 'no')

# original
# set_event

# raw without event
def test_raw_empty_event(raw):
    assert raw.has_event_str() == 'no'
    assert raw.get_event_name_list_str() == 'None'
    events, event_id = raw.get_event_list()
    assert len(events) == 0
    assert len(event_id) == 0
    events, event_id = raw.get_raw_event_list()
    assert len(events) == 0
    assert len(event_id) == 0

def test_raw_set_event_on_empty_event(raw):
    test_set_event(raw)
    # check raw event
    events, event_id = raw.get_raw_event_list()
    assert len(events) == 0
    assert len(event_id) == 0

# raw with stim event
@pytest.fixture
def mne_raw_stim():
    fs = 10
    info = mne.create_info(ch_names=['O1', 'O2', 'STI'],
                            sfreq=fs,
                            ch_types=['eeg', 'eeg', 'stim'])
    data = np.random.RandomState(0).randn(3, fs)
    stim = [0,1,0,2,0,0,3,0,0,0]
    data[2] = stim
    mne_raw = mne.io.RawArray(data, info)
    return mne_raw

@pytest.fixture
def stim_raw(mne_raw_stim):
    path = 'tests/test_data/sub-01_ses-01_task-rest_eeg.fif'    
    return Raw(path, mne_raw_stim)

def test_raw_stim_event(stim_raw):
    assert stim_raw.has_event_str() == 'yes'
    assert stim_raw.get_event_name_list_str() == '1,2,3'
    events, event_id = stim_raw.get_event_list()
    assert len(events) == 3
    assert len(event_id) == 3
    events, event_id = stim_raw.get_raw_event_list()
    assert len(events) == 3
    assert len(event_id) == 3

def test_raw_set_event_on_stim_event(stim_raw):
    test_set_event(stim_raw)
    events, event_id = stim_raw.get_raw_event_list()
    assert len(events) == 3
    assert len(event_id) == 3

# raw with annotation event
@pytest.fixture
def mne_raw_annot():
    fs = 10
    info = mne.create_info(ch_names=['O1', 'O2'],
                           sfreq=fs,
                           ch_types='eeg')
    data = np.random.RandomState(0).randn(2, fs)
    mne_raw = mne.io.RawArray(data, info)
    mne_raw.set_annotations(mne.Annotations(onset=[0.1, 0.3, 0.5],
                                            duration=[0.2, 0.2, 0.2],
                                            description=['a', 'b', 'c']))
    return mne_raw

@pytest.fixture
def annot_raw(mne_raw_annot):
    path = 'tests/test_data/sub-01_ses-01_task-rest_eeg.fif'    
    return Raw(path, mne_raw_annot)

def test_raw_annotation_event(annot_raw):
    assert annot_raw.has_event_str() == 'yes'
    assert annot_raw.get_event_name_list_str() == 'a,b,c'
    events, event_id = annot_raw.get_event_list()
    assert len(events) == 3
    assert len(event_id) == 3
    events, event_id = annot_raw.get_raw_event_list()
    assert len(events) == 3
    assert len(event_id) == 3

def test_raw_set_event_on_annotation_event(annot_raw):
    test_set_event(annot_raw)
    events, event_id = annot_raw.get_raw_event_list()
    assert len(events) == 3
    assert len(event_id) == 3

# epoch
@pytest.fixture
def mne_epoch():
    mne_raw = _generate_mne(base_fs, ['O1', 'O2'], 'eeg')

    events = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]])
    event_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    epochs = mne.Epochs(mne_raw, events, event_id, tmin=0, tmax=0.1, baseline=None, preload=True)
    return epochs

@pytest.fixture
def epoch(mne_epoch):
    path = 'tests/test_data/sub-01_ses-01_task-rest_eeg.fif'    
    return Raw(path, mne_epoch)

def test_mne_epoch_info(mne_epoch, epoch):
    assert mne_epoch == epoch.get_mne()
    assert epoch.get_tmin() == 0.0
    assert epoch.get_nchan() == 2
    assert epoch.get_sfreq() == base_fs
    assert epoch.get_filter_range() == (0, base_fs / 2)
    assert epoch.get_epochs_length() == 4
    assert epoch.get_epoch_duration() == base_fs * 0.1 + 1
    assert epoch.is_raw() == False

    assert epoch.get_row_info() == ('sub-01_ses-01_task-rest_eeg.fif', '0', '0', 2, base_fs, 4, 'yes')

def test_epoch(epoch):
    assert epoch.has_event_str() == 'yes'
    assert epoch.get_event_name_list_str() == 'a,b,c,d'
    events, event_id = epoch.get_event_list()
    assert len(events) == 4
    assert len(event_id) == 4
    events, event_id = epoch.get_raw_event_list()
    assert len(events) == 4
    assert len(event_id) == 4

def test_epoch_set_event(epoch):
    test_set_event(epoch)
    events, event_id = epoch.get_raw_event_list()
    assert len(events) == 4
    assert len(event_id) == 4

# change mne
@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_1(mne_raw_2, target, request):
    target = request.getfixturevalue(target)
    target.set_mne(mne_raw_2)
    test_mne_raw_2_info(mne_raw_2, target)

@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_2(mne_epoch, target, request):
    target = request.getfixturevalue(target)
    target.set_mne(mne_epoch)
    test_mne_epoch_info(mne_epoch, target)

@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_after_set_event_1(mne_raw_2, target, request):
    target = request.getfixturevalue(target)
    test_set_event(target)
    target.set_mne(mne_raw_2)

    assert target.has_event_str() == 'yes'
    assert target.get_event_name_list_str() == 'a,b,c,d'
    events, event_id = target.get_event_list()
    assert len(events) == 4
    assert len(event_id) == 4
    


@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_after_set_event_2(mne_epoch, target, request):
    target = request.getfixturevalue(target)
    test_set_event(target)
    target.set_mne(mne_epoch)
    
    assert target.has_event_str() == 'yes'
    assert target.get_event_name_list_str() == 'a,b,c,d'
    events, event_id = target.get_event_list()
    assert len(events) == 4
    assert len(event_id) == 4

def test_set_mne_consistency(mne_epoch, raw):
    events = np.array([[1, 0, 1], [2, 0, 2]])
    event_id = {'a': 1, 'b': 2}
    raw.set_event(events, event_id)
    with pytest.raises(AssertionError):
        raw.set_mne(mne_epoch)

@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_and_wipe_events_1(mne_raw_2, target, request):
    target = request.getfixturevalue(target)
    target.set_mne_and_wipe_events(mne_raw_2)
    test_mne_raw_2_info(mne_raw_2, target)

@pytest.mark.parametrize('target', ['raw', 'epoch'])
def test_set_mne_and_wipe_events_2(mne_epoch, target, request):
    target = request.getfixturevalue(target)
    target.set_mne_and_wipe_events(mne_epoch)
    test_mne_epoch_info(mne_epoch, target)