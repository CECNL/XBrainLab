import numpy as np
from enum import Enum
from ..option import SplitUnit
from copy import deepcopy

class DataSet:
    SEQ = 0
    def __init__(self, data_holder, config):
        self.name = ''
        self.data_holder = data_holder
        self.config = config
        self.dataset_id = DataSet.SEQ
        DataSet.SEQ += 1

        data_length = data_holder.get_data_length()
        self.remaining_mask = np.ones(data_length, dtype=bool)

        self.train_mask = np.zeros(data_length, dtype=bool)
        self.kept_training_session_list = []
        self.kept_training_subject_list = []
        self.val_mask = np.zeros(data_length, dtype=bool)
        self.test_mask = np.zeros(data_length, dtype=bool)
        self.is_selected = True
    
    # data splitting
    ## getter
    ### info
    def get_data_holder(self):
        return self.data_holder

    def get_name(self):
        return str(self.dataset_id) + '-' + self.name

    ### mask
    def get_remaining_mask(self):
        return self.remaining_mask.copy()

    def get_all_trail_numbers(self):
        train_number = sum(self.train_mask)
        val_number = sum(self.val_mask)
        test_number = sum(self.test_mask)
        return train_number, val_number, test_number
    
    def has_set_empty(self):
        train_number, val_number, test_number = self.get_all_trail_numbers()
        return train_number == 0 or val_number == 0 or test_number == 0

    def get_treeview_row_info(self):
        train_number, val_number, test_number = self.get_all_trail_numbers()
        selected = 'O' if self.is_selected else 'X'
        name = self.get_name()
        return selected, name, train_number, val_number, test_number

    ## setter
    def set_selection(self, select):
        self.is_selected = select

    def set_name(self, name):
        self.name = name
    
    ## picker
    def discard(self, mask):
        self.remaining_mask &= np.logical_not(mask)

    def set_remaining_by_subject_idx(self, idx):
        self.remaining_mask = self.data_holder.pick_subject_mask_by_idx(idx)

    ## keep from validation
    def kept_training_session(self, mask):
        self.kept_training_session_list = np.unique(self.data_holder.session[mask])

    def kept_training_subject(self, mask):
        self.kept_training_subject_list = np.unique(self.data_holder.subject[mask])
    
    ## set result
    def set_test(self, mask):
        self.test_mask = mask.copy()
        self.remaining_mask &= np.logical_not(mask)
        for kept_training_session in self.kept_training_session_list:
            target = self.data_holder.get_session_list() == kept_training_session
            self.train_mask |= (self.remaining_mask & target)
            self.remaining_mask &= np.logical_not(target)
        for kept_training_subject in self.kept_training_subject_list:
            target = self.data_holder.subject == kept_training_subject
            self.train_mask |= (self.remaining_mask & target)
            self.remaining_mask &= np.logical_not(target)
    
    def set_val(self, mask):
        self.val_mask = mask.copy()
        self.remaining_mask &= np.logical_not(mask)

    def set_train(self):
        self.train_mask |= self.remaining_mask
        self.remaining_mask &= False

    ## filter
    def intersection_with_subject_by_idx(self, mask, idx):
        return mask & self.data_holder.pick_subject_mask_by_idx(idx)

    # train
    def get_training_data(self):
        X = self.data_holder.get_data()[self.train_mask]
        y = self.data_holder.get_label_list()[self.train_mask]   
        return X, y

    def get_val_data(self):
        X = self.data_holder.get_data()[self.val_mask]
        y = self.data_holder.get_label_list()[self.val_mask]
        return X, y

    def get_test_data(self):
        X = self.data_holder.get_data()[self.test_mask]
        y = self.data_holder.get_label_list()[self.test_mask]
        return X, y
    
    # get data len
    def get_train_len(self):
        return sum(self.train_mask)

    def get_val_len(self):
        return sum(self.val_mask)

    def get_test_len(self):
        return sum(self.test_mask)
        