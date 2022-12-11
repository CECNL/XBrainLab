import numpy as np
from enum import Enum
from copy import deepcopy
from .option import SplitUnit
from .epochs import Epochs

from ..utils import validate_type

class Dataset:
    SEQ = 0
    def __init__(self, epoch_data, config):
        validate_type(epoch_data, Epochs ,"epoch_data")
        self.name = ''
        self.epoch_data = epoch_data
        self.config = config
        self.dataset_id = Dataset.SEQ
        Dataset.SEQ += 1

        data_length = epoch_data.get_data_length()
        self.remaining_mask = np.ones(data_length, dtype=bool)

        self.train_mask = np.zeros(data_length, dtype=bool)
        self.val_mask = np.zeros(data_length, dtype=bool)
        self.test_mask = np.zeros(data_length, dtype=bool)
        self.is_selected = True
    
    # data splitting
    ## getter
    ### info
    def get_epoch_data(self):
        return self.epoch_data

    def get_name(self):
        return str(self.dataset_id) + '-' + self.name

    def get_ori_name(self):
        return self.name

    def get_all_trial_numbers(self):
        train_number = sum(self.train_mask)
        val_number = sum(self.val_mask)
        test_number = sum(self.test_mask)
        return train_number, val_number, test_number
    
    def get_treeview_row_info(self):
        train_number, val_number, test_number = self.get_all_trial_numbers()
        selected = 'O' if self.is_selected else 'X'
        name = self.get_name()
        return selected, name, train_number, val_number, test_number

    ## setter
    def set_selection(self, select):
        self.is_selected = select

    def set_name(self, name):
        self.name = name
    
    def has_set_empty(self):
        train_number, val_number, test_number = self.get_all_trial_numbers()
        return train_number == 0 or val_number == 0 or test_number == 0

    ### mask
    def get_remaining_mask(self):
        return self.remaining_mask.copy()

    ## picker
    def discard_remaining_mask(self, mask):
        self.remaining_mask &= np.logical_not(mask)

    def set_remaining_by_subject_idx(self, idx):
        self.remaining_mask = self.epoch_data.pick_subject_mask_by_idx(idx)

    ## set result
    def set_test(self, mask):
        self.test_mask = mask.copy()
        self.remaining_mask &= np.logical_not(mask)
    
    def set_val(self, mask):
        self.val_mask = mask.copy()
        self.remaining_mask &= np.logical_not(mask)

    def set_remaining_to_train(self):
        self.train_mask |= self.remaining_mask
        self.remaining_mask &= False

    ## filter
    def intersection_with_subject_by_idx(self, mask, idx):
        return mask & self.epoch_data.pick_subject_mask_by_idx(idx)

    # train
    def get_training_data(self):
        X = self.epoch_data.get_data()[self.train_mask]
        y = self.epoch_data.get_label_list()[self.train_mask]   
        return X, y

    def get_val_data(self):
        X = self.epoch_data.get_data()[self.val_mask]
        y = self.epoch_data.get_label_list()[self.val_mask]
        return X, y

    def get_test_data(self):
        X = self.epoch_data.get_data()[self.test_mask]
        y = self.epoch_data.get_label_list()[self.test_mask]
        return X, y
    
    # get data len
    def get_train_len(self):
        return sum(self.train_mask)

    def get_val_len(self):
        return sum(self.val_mask)

    def get_test_len(self):
        return sum(self.test_mask)
    