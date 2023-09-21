import mne
import numpy as np
import pytest

from XBrainLab.dataset import Epochs, SplitUnit
from XBrainLab.load_data import Raw

epoch_duration = 3
n_class = 2
n_trial = 3
fs = 5
block_size = n_class * n_trial
subject_list = ['1', '2', '3']
session_list = ['1', '2']
event_id = {'c1': 0, 'c2': 1}
ch_names=['O1', 'O2']

@pytest.fixture
def preprocessed_data_list():
    events = np.zeros((n_class * n_trial, 3), dtype=int)
    events[:, 0] = np.arange(events.shape[0])
    events[:, 2] = np.arange(n_class).repeat(n_trial)

    ch_types = 'eeg'

    result = []
    for subject in subject_list:
        for session in session_list:
            base = int(subject) * 100000 + int(session) * 1000
            info = mne.create_info(ch_names=ch_names,
                           sfreq=fs,
                           ch_types=ch_types)
            data = np.zeros((len(events), len(ch_names), epoch_duration * fs))
            for i in range(len(events)):
                data[i, :, :] = base + events[i, 0]
            epochs = mne.EpochsArray(
                data, info, events=events, tmin=0, event_id=event_id
            )
            raw = Raw(f"test/sub-{subject}_ses-{session}.fif", epochs)
            raw.set_subject_name(subject)
            raw.set_session_name(session)
            result.append(raw)
    return result

@pytest.fixture
def epochs(preprocessed_data_list):
    return Epochs(preprocessed_data_list)

@pytest.fixture
def full_filter_preview_mask(epochs):
    mask = np.ones(block_size * len(subject_list) * len(session_list), dtype=bool)
    filter_preview_mask = epochs._generate_mask_target(mask)
    return filter_preview_mask

def test_epochs_args_error():
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    data = np.zeros((2, 5))
    raw = Raw('test/sub-01_ses-01_task-rest_eeg.fif', mne.io.RawArray(data, info))
    with pytest.raises(ValueError, match=".*of type epoch."):
        Epochs([raw])

def test_epochs_subject_attributes(epochs):
    for subject in range(len(subject_list)):
        for session in range(len(session_list)):
            i = subject * len(session_list) + session
            assert (
                epochs.get_subject_list()[
                    i * block_size: (i + 1) * block_size
                ] ==
                subject
            ).all()
    assert set(epochs.get_subject_map().values()) == set(subject_list)
    assert epochs.get_subject_index_list() == list(range(len(subject_list)))

def test_epochs_session_attributes(epochs):
    for subject in range(len(subject_list)):
        for session in range(len(session_list)):
            i = subject * len(session_list) + session
            assert (
                epochs.get_session_list()[
                    i * block_size: (i + 1) * block_size
                ] ==
                session
            ).all()
    assert set(epochs.get_session_map().values()) == set(session_list)

def test_epochs_label_attributes(epochs):
    for subject in range(len(subject_list)):
        for session in range(len(session_list)):
            i = subject * len(session_list) + session
            assert (
                epochs.get_label_list()[
                    i * block_size: (i + 1) * block_size
                ] ==
                np.arange(n_class).repeat(n_trial)
            ).all()
    assert set(epochs.get_label_map().values()) == set(event_id.keys())

def test_epochs_copy(epochs):
    epochs_copy = epochs.copy()
    old_sfreq = epochs.sfreq
    epochs_copy.sfreq = -1
    assert epochs.sfreq == old_sfreq

def test_epochs_get_by_mask(epochs):
    mask = np.zeros(block_size * len(subject_list) * len(session_list), dtype=bool)
    mask[block_size:block_size * 2] = True
    assert np.allclose(epochs.get_subject_list_by_mask(mask),
                       np.array([0] * block_size))
    assert np.allclose(epochs.get_session_list_by_mask(mask),
                       np.array([1] * block_size))
    assert np.allclose(epochs.get_label_list_by_mask(mask),
                       np.arange(n_class).repeat(n_trial))
    assert np.allclose(epochs.get_idx_list_by_mask(mask),
                       np.arange(block_size))

    mask &= False
    count = block_size * len(session_list)
    mask[count: count+block_size*len(session_list)] = True
    assert (epochs.pick_subject_mask_by_idx(1) == mask).all()

def test_epochs_get_by_index(epochs):
    for idx, name in enumerate(subject_list):
        assert epochs.get_subject_name(idx) == name

    for idx, name in enumerate(session_list):
        assert epochs.get_session_name(idx) == name

    for idx, name in enumerate(event_id):
        assert epochs.get_label_name(idx) == name

def test_epochs_info(epochs):
    assert (
        epochs.get_data_length() ==
        block_size * len(subject_list) * len(session_list)
    )
    assert epochs.get_model_args() == {
        'n_classes': len(event_id),
        'channels': len(ch_names),
        'samples': epoch_duration * fs,
        'sfreq': fs
    }
    assert (
        epochs.get_data().shape ==
        (block_size * len(subject_list) * len(session_list),
         len(ch_names), epoch_duration * fs)
    )
    assert epochs.get_label_number() == len(ch_names)
    assert epochs.get_channel_names() == ch_names
    assert np.isclose(epochs.get_epoch_duration(), epoch_duration)

def test_epochs_set_channel(epochs):
    new_ch_names = ['O1', 'O2', 'O3']
    channel_position = np.random.rand(3, 3).tolist()
    epochs.set_channels(new_ch_names, channel_position)
    assert epochs.get_channel_names() == new_ch_names
    assert epochs.get_montage_position() == channel_position

def test_epochs_generate_mask_target(full_filter_preview_mask):
    for label_idx in range(len(event_id)):
        assert label_idx in full_filter_preview_mask
        for subject_idx in range(len(subject_list)):
            assert subject_idx in full_filter_preview_mask[label_idx]
            for session_idx in range(len(session_list)):
                assert session_idx in full_filter_preview_mask[label_idx][subject_idx]
                target_filter_mask, counter = (
                    full_filter_preview_mask[label_idx][subject_idx][session_idx]
                )
                assert counter == 0
                assert (
                    target_filter_mask.shape ==
                    (block_size * len(session_list) * len(subject_list),)
                )
                assert sum(target_filter_mask) == n_trial

def test_epochs_generate_mask_target_partial(epochs):
    mask = np.ones(block_size * len(subject_list) * len(session_list), dtype=bool)
    mask[:block_size * len(session_list)] = False
    filter_preview_mask = epochs._generate_mask_target(mask)
    for label_idx in range(len(event_id)):
        assert label_idx in filter_preview_mask
        for subject_idx in range(len(subject_list)):
            assert subject_idx in filter_preview_mask[label_idx]
            for session_idx in range(len(session_list)):
                assert session_idx in filter_preview_mask[label_idx][subject_idx]
                target_filter_mask, counter = \
                    filter_preview_mask[label_idx][subject_idx][session_idx]
                assert counter == 0
                assert target_filter_mask.shape == (
                    block_size * len(session_list) * len(subject_list),
                )
                if subject_idx == 0:
                    assert sum(target_filter_mask) == 0
                else:
                    assert sum(target_filter_mask) == n_trial

def test_epochs_get_filtered_mask_pair(epochs, full_filter_preview_mask):
    for label_idx in range(len(event_id)):
        for subject_idx in range(len(subject_list)):
            for session_idx in range(len(session_list)):
                full_filter_preview_mask[label_idx][subject_idx][session_idx][1] = 1
    target_session = 1
    target_label = 0
    target_subject = 2
    full_filter_preview_mask[target_label][target_subject][target_session][1] = 0
    expect = full_filter_preview_mask[target_label][target_subject][target_session]
    result = epochs._get_filtered_mask_pair(full_filter_preview_mask)
    assert (expect[0] == result[0]).all()
    assert expect[1] == result[1]

def test_epochs_update_mask_target(epochs, full_filter_preview_mask):
    pos = np.zeros(block_size * len(subject_list) * len(session_list), dtype=bool)
    pos[:block_size * len(session_list)] = True
    epochs._update_mask_target(full_filter_preview_mask, pos)
    for label_idx in range(len(event_id)):
        for subject_idx in range(len(subject_list)):
            for session_idx in range(len(session_list)):
                target = full_filter_preview_mask[label_idx][subject_idx][session_idx]
                if subject_idx == 0:
                    assert (target[1] == n_trial)
                    assert sum(target[0]) == 0
                else:
                    assert target[1] == 0
                    assert sum(target[0]) == n_trial

def _test_epochs_get_real_num_param():
    params = [
        (1, SplitUnit.NUMBER, 1),
        (4, SplitUnit.NUMBER, 4),
        (200, SplitUnit.NUMBER, 4)
    ]
    params += [
        (i, SplitUnit.RATIO, int(i * 4))
        for i in np.arange(0, 1, 0.1)
    ]
    return params

@pytest.mark.parametrize(
        'value, split_unit, expected', _test_epochs_get_real_num_param()
    )
@pytest.mark.parametrize('mask, clean_mask', [
    (np.ones(16, dtype=bool), None),
    (np.zeros(16, dtype=bool), np.ones(16, dtype=bool))
])
def test_epochs_get_real_num(epochs, value, split_unit, expected, mask, clean_mask):
    target_type = np.arange(4).repeat(4)
    group_idx = 0
    assert (
        expected ==
        epochs._get_real_num(
            target_type, value, split_unit, mask, clean_mask, group_idx
        )
    )

def _test_epochs_get_real_num_partial_param():
    params = [
        (1, SplitUnit.NUMBER, 1),
        (4, SplitUnit.NUMBER, 3),
        (200, SplitUnit.NUMBER, 3)
    ]
    params += [
        (i, SplitUnit.RATIO, int(i * 3))
        for i in np.arange(0, 1, 0.1)
    ]
    return params

@pytest.mark.parametrize(
    'value, split_unit, expected',
    _test_epochs_get_real_num_partial_param()
)
def test_epochs_get_real_num_partial(epochs, value, split_unit, expected):
    target_type = np.arange(4).repeat(4)
    group_idx = 0
    mask = np.ones(16, dtype=bool)
    clean_mask = None
    mask[:4] = False
    assert expected == epochs._get_real_num(
        target_type, value, split_unit, mask, clean_mask, group_idx
    )


@pytest.mark.parametrize('value, group_idx, expected, is_partial', [
    (1, 0, 4, 0),
    (2, 0, 2, 0),
    (2, 1, 2, 0),
    (3, 0, 2, 0),
    (3, 1, 1, 0),
    (3, 2, 1, 0),

    (1, 0, 3, 1),
    (2, 0, 2, 1),
    (2, 1, 1, 1),
    (3, 0, 1, 1),
    (3, 1, 1, 1),
    (3, 2, 1, 1)
])
def test_epochs_get_real_num_k_fold(epochs, value, group_idx, expected, is_partial):
    target_type = np.arange(4).repeat(4)
    split_unit = SplitUnit.KFOLD
    mask = np.ones(16, dtype=bool)
    if is_partial:
        mask[:4] = False
    clean_mask = None

    assert expected == epochs._get_real_num(
        target_type, value, split_unit, mask, clean_mask, group_idx
    )


def test_epochs_get_real_num_not_implemented(epochs):
    with pytest.raises(NotImplementedError):
        epochs._get_real_num(np.arange(4), 1, 'test', np.ones(4, dtype=bool), None, 0)

@pytest.mark.parametrize('selected_num', np.arange(block_size + 2))
@pytest.mark.parametrize('is_partial', [False, True])
def test_epochs_pick(mocker, epochs, selected_num, is_partial):
    target_type = np.arange(block_size).repeat(len(subject_list) * len(session_list))
    mask = np.ones(len(target_type), dtype=bool)
    real_block_size = block_size
    if is_partial:
        mask[:block_size] = False
        real_block_size -= 1
    old_mask = mask.copy()
    clean_mask = None
    value = 0
    split_unit = 0
    group_idx = 0
    mocker.patch.object(epochs, '_get_real_num', return_value=selected_num)

    ret, new_mask = epochs._pick(
        target_type, mask, clean_mask, value, split_unit, group_idx
    )

    assert (new_mask == mask).all()
    if is_partial:
        assert sum(ret) == sum(old_mask & ret)
    selected = target_type[ret]
    non_selected = target_type[np.logical_not(ret)]

    selected_idx_list = np.unique(selected)
    non_selected_idx_list = np.unique(non_selected)
    if selected_num > real_block_size:
        assert len(selected_idx_list) == real_block_size
    else:
        assert len(selected_idx_list) == selected_num

    assert len(set(selected_idx_list).intersection(set(non_selected_idx_list))) == 0

def test_epochs_pick_manual(epochs):
    target_type = np.arange(block_size).repeat(len(subject_list) * len(session_list))
    mask = np.ones(len(target_type), dtype=bool)
    value = [3, 5]
    result, new_mask = epochs._pick_manual(target_type, mask, value)

    assert (new_mask == mask).all()
    selected = target_type[result]
    non_selected = target_type[np.logical_not(result)]
    selected_idx_list = np.unique(selected)
    non_selected_idx_list = np.unique(non_selected)

    assert set(selected_idx_list) == set(value)
    assert len(set(selected_idx_list).intersection(set(non_selected_idx_list))) == 0

@pytest.mark.parametrize('func_name, target_type_name', [
    ('pick_subject', 'get_subject_list'),
    ('pick_session', 'get_session_list')
])
@pytest.mark.parametrize('split_unit, is_manual', [
    (SplitUnit.MANUAL, True),
    (SplitUnit.NUMBER, False),
    (SplitUnit.KFOLD, False),
    (SplitUnit.RATIO, False),
])
def test_epochs_pick_by_wrapper(
    mocker, epochs, func_name, split_unit, is_manual, target_type_name
):
    pick_mock = mocker.patch.object(epochs, '_pick')
    manual_mock = mocker.patch.object(epochs, '_pick_manual')

    target_type = getattr(epochs, target_type_name)()
    mask = np.random.randint(0, 2, size=len(target_type), dtype=bool)
    clean_mask = None
    group_idx = 5
    value = [1, 2, 3]
    # call func_name of epochs
    func = getattr(epochs, func_name)

    func(mask, clean_mask, value, split_unit, group_idx)
    if is_manual:
        manual_mock.assert_called_once()
        pick_mock.assert_not_called()
        (_target_type, _mask, _value), _ = manual_mock.call_args
        assert (_target_type == target_type).all()
        assert (_mask == mask).all()
        assert _value == value
    else:
        pick_mock.assert_called_once()
        manual_mock.assert_not_called()
        (_target_type, _mask, _clean_mask, _value, _split_unit, _group_idx), _ = \
            pick_mock.call_args
        assert (_target_type == target_type).all()
        assert (_mask == mask).all()
        assert _clean_mask == clean_mask
        assert _value == value
        assert _split_unit == split_unit
        assert _group_idx == group_idx

def test_epochs_pick_manual_trial(epochs):
    mask = np.ones(block_size * len(subject_list) * len(session_list), dtype=bool)
    clean_mask = None
    value = np.random.randint(
        0, 2, size=block_size * len(subject_list) * len(session_list), dtype=bool
    )

    result, _ = epochs.pick_trial(mask, clean_mask, value, SplitUnit.MANUAL, 0)
    assert (result == value).all()

def _generate_expected_epochs_pick_by_trial_param(count, is_partial):
    expected = np.zeros(block_size * len(subject_list) * len(session_list), dtype=bool)
    for repeat in range(n_trial):
        for sess in range(len(session_list)):
            for sub in range(len(subject_list)):
                for label in range(n_class):
                    if count <= 0:
                        break
                    if is_partial and sub == 0:
                        continue
                    expected[(n_trial - repeat - 1) +
                              label * n_trial +
                              sess * block_size +
                              sub * block_size * len(session_list)] = True
                    count -= 1
    return expected

def _test_epochs_pick_by_trial_partial_param():
    is_partial = True
    total_count = block_size * (len(subject_list) - 1) * len(session_list)
    params = [
        (
            SplitUnit.NUMBER, i,
            _generate_expected_epochs_pick_by_trial_param(i, is_partial), 0, is_partial
        )
        for i in range(block_size * len(subject_list) * len(session_list) + 2)
    ]
    params += [
        (
            SplitUnit.KFOLD, 1,
            _generate_expected_epochs_pick_by_trial_param(total_count, is_partial),
            0, is_partial
        )
    ]
    for i in range(4):
        params.append(
            (
                SplitUnit.KFOLD, 10,
                _generate_expected_epochs_pick_by_trial_param(3, is_partial),
                i, is_partial
            )
        )
    for i in range(2):
        params.append(
            (
                SplitUnit.KFOLD, 10,
                _generate_expected_epochs_pick_by_trial_param(2, is_partial),
                4 + i, is_partial
            ),
        )

    for i in np.arange(0, 1, 0.1):
        count = int(i * total_count)
        expected = _generate_expected_epochs_pick_by_trial_param(count, is_partial)
        params.append((SplitUnit.RATIO, i, expected, 0, is_partial))
    return params

def _test_epochs_pick_by_trial_param():
    params = []
    is_partial = False
    total_count = block_size * len(subject_list) * len(session_list)
    params += [
        (
            SplitUnit.NUMBER, i,
            _generate_expected_epochs_pick_by_trial_param(i, is_partial),
            0, is_partial
        )
        for i in range(block_size * len(subject_list) * len(session_list) + 2)
    ]
    params += [
        (
            SplitUnit.KFOLD, 1,
            _generate_expected_epochs_pick_by_trial_param(total_count, is_partial),
            0, is_partial
        ),
        (
            SplitUnit.KFOLD, 10,
            _generate_expected_epochs_pick_by_trial_param(4, is_partial),
            0, is_partial
        ),
        (
            SplitUnit.KFOLD, 10,
            _generate_expected_epochs_pick_by_trial_param(4, is_partial),
            1, is_partial
        ),
        (
            SplitUnit.KFOLD, 10,
            _generate_expected_epochs_pick_by_trial_param(4, is_partial),
            2, is_partial
        ),
        (
            SplitUnit.KFOLD, 10,
            _generate_expected_epochs_pick_by_trial_param(3, is_partial),
            6, is_partial
        ),
    ]
    for i in np.arange(0, 1, 0.1):
        count = int(i * total_count)
        expected = _generate_expected_epochs_pick_by_trial_param(count, is_partial)
        params.append((SplitUnit.RATIO, i, expected, 0, is_partial))
    params += _test_epochs_pick_by_trial_partial_param()
    return params

@pytest.mark.parametrize(
        'split_unit, value, expected, group_idx, is_partial',
        _test_epochs_pick_by_trial_param()
    )
@pytest.mark.parametrize('clean_mask', [
    None, np.ones(block_size * len(subject_list) * len(session_list), dtype=bool)
])
def test_epochs_pick_by_trial(
    epochs, clean_mask,
    split_unit, value, expected,
    group_idx, is_partial
):
    mask = np.ones(block_size * len(subject_list) * len(session_list), dtype=bool)
    if is_partial:
        if clean_mask is not None:
            clean_mask[:block_size  * len(session_list)] = False
        mask[:block_size  * len(session_list)] = False
    result, new_mask = epochs.pick_trial(
        mask, clean_mask, value, split_unit, group_idx
    )
    assert (new_mask == mask).all()
    assert sum(result) == sum(expected)
    assert (result == expected).all()

def test_epochs_pick_by_trial_not_implemented(epochs):
    with pytest.raises(NotImplementedError):
        epochs.pick_trial(np.arange(4), None, 1, 'test', 0)
