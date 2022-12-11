import numpy as np

from .option import TrainingType, SplitByType, ValSplitByType, SplitUnit
from .dataset import Dataset
from . import DataSplittingConfig, Epochs

from ..utils import validate_type

class DatasetGenerator:
    def __init__(self, epoch_data, config, datasets=None):
        validate_type(epoch_data, Epochs ,"epoch_data")
        validate_type(config, DataSplittingConfig ,"config")
        if datasets is None:
            datasets = []
        else:
            assert len(datasets) == 0
        self.epoch_data = epoch_data
        self.config = config
        self.test_splitter_list = config.test_splitter_list
        self.val_splitter_list = config.val_splitter_list
        
        self.datasets = datasets
        self.interrupted = False
        self.preview_failed = False

    def handle_IND(self):
        for subject_idx in range(len(self.epoch_data.get_subject_index_list())):
            name_prefix = f"Subject-{self.epoch_data.get_subject_name(subject_idx)}"
            def hook(dataset):
                dataset.set_remaining_by_subject_idx(subject_idx)
            self.handle(name_prefix, hook)
    
    def handle_FULL(self):
        name_prefix = "Group"
        self.handle(name_prefix)
    
    def split_test(self, dataset, group_idx, mask, clean_mask):
        idx = 0
        next_mask = mask.copy()
        for test_splitter in self.test_splitter_list:
            if test_splitter.is_option:
                if self.interrupted:
                    raise KeyboardInterrupt
                if not test_splitter.is_valid():
                    self.preview_failed = True
                    raise ValueError("Preview failed")
                # session
                if test_splitter.split_type == SplitByType.SESSION or test_splitter.split_type == SplitByType.SESSION_IND:
                    split_func = self.epoch_data.pick_session
                # label
                elif test_splitter.split_type == SplitByType.TRIAL or test_splitter.split_type == SplitByType.TRIAL_IND:
                    split_func = self.epoch_data.pick_trial
                # subject
                elif test_splitter.split_type == SplitByType.SUBJECT or test_splitter.split_type == SplitByType.SUBJECT_IND:
                    split_func = self.epoch_data.pick_subject
                
                mask, excluded = split_func(
                    mask=mask, clean_mask=clean_mask, 
                    value=test_splitter.get_value(), split_unit=test_splitter.get_split_unit(), group_idx=group_idx)
                # save for next cross validation
                if idx == 0:
                    next_mask = excluded.copy()
                    # restore previous cross validation part
                    excluded = clean_mask & np.logical_not(mask)
                    clean_mask = None
                if not mask.any():
                    break
                # independent
                if test_splitter.split_type == SplitByType.SESSION_IND or test_splitter.split_type == SplitByType.TRIAL_IND or test_splitter.split_type == SplitByType.SUBJECT_IND:
                    dataset.discard_remaining_mask(excluded)
                idx += 1
        if idx > 0:
            dataset.set_test(mask)
        else:
            next_mask &= False
        return next_mask

    def split_validate(self, dataset, group_idx, clean_mask):
        mask = dataset.get_remaining_mask()
        idx = 0
        for val_splitter in self.val_splitter_list:
            if val_splitter.is_option:
                # check job interrupt
                if self.interrupted:
                    raise KeyboardInterrupt
                if not val_splitter.is_valid():
                    self.preview_failed = True
                    return
                # session
                if val_splitter.split_type == ValSplitByType.SESSION:
                    split_func = self.epoch_data.pick_session
                # label
                elif val_splitter.split_type == ValSplitByType.TRIAL:
                    split_func = self.epoch_data.pick_trial
                # subject
                elif val_splitter.split_type == ValSplitByType.SUBJECT:
                    split_func = self.epoch_data.pick_subject
                mask, excluded = split_func(mask=mask, clean_mask=None, 
                    value=val_splitter.get_value(), split_unit=val_splitter.get_split_unit(), group_idx=group_idx)
                idx += 1
        if idx > 0:
            dataset.set_val(mask)

    def handle(self, name_prefix, dataset_hook=None):
        group_idx = 0
        remaining_mask = None
        while (remaining_mask is None) or (self.config.is_cross_validation and remaining_mask.any()):
            dataset = Dataset(self.epoch_data, self.config)
            dataset.set_name(f"{name_prefix}_{group_idx}")
            if dataset_hook:
                dataset_hook(dataset)
            clean_mask = dataset.get_remaining_mask()
            
            if remaining_mask is None:
                mask = dataset.get_remaining_mask()
            else:
                mask = remaining_mask
            remaining_mask = self.split_test(dataset, group_idx, mask, clean_mask)
            self.split_validate(dataset, group_idx, clean_mask)
            dataset.set_remaining_to_train()
            
            self.datasets.append(dataset)
            group_idx += 1

    def generate(self):
        if self.datasets:
            return self.datasets
        Dataset.SEQ = 0
        # individual scheme
        if self.config.train_type == TrainingType.IND:
            self.handle_IND()
        elif self.config.train_type == TrainingType.FULL:
            self.handle_FULL()
        else:
            raise NotImplementedError
        
        if len(self.datasets) == 0:
            self.preview_failed = True
            raise ValueError
        
        return self.datasets

    def set_interrupt(self):
        self.interrupted = True

    def prepare_reuslt(self):
        if not self.datasets:
            self.generate()
        while True:
            done = True
            for i in range(len(self.datasets)):
                if not self.datasets[i].is_selected:
                    del self.datasets[i]
                    done = False
                    break
            if done:
                break
        
        # check if dataset is empty
        if len(self.datasets) == 0:
            raise ValueError('No valid dataset is generated')
        return self.datasets
    
    def apply(self, study):
        from ..lab import XBrainLab
        validate_type(study, XBrainLab, 'study')
        self.prepare_reuslt()
        study.set_datasets(self.datasets)
