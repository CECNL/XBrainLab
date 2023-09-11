from XBrainLab.dataset import Dataset, DataSplittingConfig, TrainingType
from .test_epochs import epochs, preprocessed_data_list, block_size, session_list, n_class, n_trial

import pytest
import numpy as np

def test_dataset(epochs):
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    
    assert dataset.get_epoch_data() == epochs
    assert dataset.get_name() == '0-'

    dataset = Dataset(epochs, config)
    dataset.set_name('test')
    assert dataset.get_name() == '1-test'
    assert dataset.get_ori_name() == 'test'

    assert dataset.get_all_trial_numbers() == (0, 0, 0) # TODO
    assert dataset.get_treeview_row_info() == ('O', '1-test', 0, 0, 0)
    dataset.set_selection(False)
    assert dataset.get_treeview_row_info() == ('X', '1-test', 0, 0, 0)
    assert dataset.has_set_empty() == True

    X, y = dataset.get_training_data() # TODO
    assert len(X) == 0
    assert len(y) == 0
    X, y = dataset.get_val_data() # TODO
    assert len(X) == 0
    assert len(y) == 0
    X, y = dataset.get_test_data() # TODO
    assert len(X) == 0
    assert len(y) == 0

    assert dataset.get_remaining_mask().all() == True
    assert dataset.get_train_len() == 0
    assert dataset.get_val_len() == 0
    assert dataset.get_test_len() == 0

def test_dataset_set_test_mask(epochs):
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    
    total = epochs.get_data_length()
    # set test
    mask[:3] = True
    dataset.set_test(mask)
    assert dataset.has_set_empty() == True
    assert sum(dataset.get_remaining_mask()) == (total - 3)

    # set val
    mask[3:9] = True
    dataset.set_val(mask)
    assert dataset.has_set_empty() == True
    assert sum(dataset.get_remaining_mask()) == (total - 9)
    # set train
    dataset.set_remaining_to_train()
    assert dataset.has_set_empty() == False
    np.logical_not(dataset.get_remaining_mask()).all() == True

    assert dataset.get_train_len() == (total - 9)
    assert dataset.get_val_len() == 6
    assert dataset.get_test_len() == 3
    assert dataset.get_all_trial_numbers() == (total - 9, 6, 3)
    _, _, train_number, val_number, test_number = dataset.get_treeview_row_info()
    assert (train_number, val_number, test_number) == (total - 9, 6, 3)

def test_dataset_discard(epochs):
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    mask[:5] = True
    dataset.discard_remaining_mask(mask)
    assert dataset.get_remaining_mask()[:5].any() == False
    assert dataset.get_remaining_mask()[5:].all() == True

def test_dataset_set_remaining_by_subject_idx(epochs):
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    dataset.set_remaining_by_subject_idx(0)
    assert dataset.get_remaining_mask()[:block_size * len(session_list)].all() == True
    assert dataset.get_remaining_mask()[block_size * len(session_list):].any() == False


subject_count = block_size * len(session_list)
half_subject_count = subject_count // 2
@pytest.mark.parametrize('start, end', [
    (subject_count, subject_count * 2),
    (half_subject_count, subject_count),
    (0, subject_count * 2),
    (subject_count, subject_count * 2)
])
def test_dataset_intersection_with_subject_by_idx(epochs, start, end):    
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    mask = np.zeros(epochs.get_data_length(), dtype=bool)

    mask[start:end] = True
    result = dataset.intersection_with_subject_by_idx(mask, 0)
    assert (result[:subject_count] == mask[:subject_count]).all() == True
    assert (result[subject_count:] == False).all() == True


def test_dataset_get_data(epochs):
    config = DataSplittingConfig(TrainingType.IND, False, [], [])
    dataset = Dataset(epochs, config)
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    mask[:subject_count] = True
    dataset.set_test(mask)
    
    mask &= False
    mask[subject_count:subject_count * 2] = True
    dataset.set_val(mask)

    mask &= False
    mask[subject_count * 3:] = True
    dataset.discard_remaining_mask(mask)
    dataset.set_remaining_to_train()
    
    X, y = dataset.get_training_data()
    assert (X // 100000 == 3).all() == True
    assert np.array_equal(y, np.tile(np.arange(n_class).repeat(n_trial), len(session_list))) == True

    X, y = dataset.get_val_data()
    assert (X // 100000 == 2).all() == True
    assert np.array_equal(y, np.tile(np.arange(n_class).repeat(n_trial), len(session_list))) == True

    X, y = dataset.get_test_data()
    assert (X // 100000 == 1).all() == True
    assert np.array_equal(y, np.tile(np.arange(n_class).repeat(n_trial), len(session_list))) == True
    


