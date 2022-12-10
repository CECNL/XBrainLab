import numpy as np

from .option import TrainingType, SplitByType, ValSplitByType, SplitUnit
from .dataset import Dataset
from . import DataSplittingConfig, Epochs

from ..utils import validate_type

class DatasetGenerator:
    def __init__(self, epoch_data, config, datasets=[]):
        validate_type(epoch_data, Epochs ,"epoch_data")
        validate_type(config, DataSplittingConfig ,"config")
        self.epoch_data = epoch_data
        self.config = config
        self.test_splitter_list = config.test_splitter_list
        self.val_splitter_list = config.val_splitter_list
        
        self.datasets = datasets
        self.interrupted = False
        self.preview_failed = False

    def generate(self):
        if self.datasets:
            return self.datasets
        Dataset.SEQ = 0
        # for loop for individual scheme
        # break at the end if not individual scheme
        for subject_idx in range(len(self.epoch_data.get_subject_index_list())):
            group_idx = 0
            # parms for cross validation
            # break at the end if not cross validation
            has_next = True
            ref_mask = None
            ref_exclude = None
            while has_next:
                # check job interrupt
                if self.interrupted:
                    raise KeyboardInterrupt
                dataset = Dataset(self.epoch_data, self.config)
                dataset.set_name(f"Group {group_idx}")
                # set name to subject-xxx for individual scheme
                if self.config.train_type == TrainingType.IND:
                    dataset.set_remaining_by_subject_idx(subject_idx)
                    if group_idx > 0:
                        dataset.set_name(f"Subject {self.epoch_data.get_subject_name(subject_idx)}-{group_idx}")
                    else:
                        dataset.set_name(f"Subject {self.epoch_data.get_subject_name(subject_idx)}")
                # get reference mask
                if ref_mask is None:
                    mask = dataset.get_remaining_mask()
                    ref_exclude = np.logical_not(mask)
                else:
                    mask = ref_mask
                    ref_exclude = dataset.get_remaining_mask() & np.logical_not(ref_mask)
                # filter out non-target subjects for individual scheme
                if self.config.train_type == TrainingType.IND:
                    ref_exclude = dataset.intersection_with_subject_by_idx(ref_exclude, subject_idx)
                # split for test
                idx = 0
                for test_splitter in self.test_splitter_list:
                    if test_splitter.is_option:
                        if self.interrupted:
                            raise KeyboardInterrupt
                        if not test_splitter.is_valid():
                            self.preview_failed = True
                            raise ValueError("Preview failed")
                        # session
                        if test_splitter.split_type == SplitByType.SESSION or test_splitter.split_type == SplitByType.SESSION_IND:
                            mask, exclude = self.epoch_data.pick_session(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.SESSION_IND:
                                dataset.discard(exclude)
                        # label
                        elif test_splitter.split_type == SplitByType.TRIAL or test_splitter.split_type == SplitByType.TRIAL_IND:
                            mask, exclude = self.epoch_data.pick_trial(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.TRIAL_IND:
                                dataset.discard(exclude)
                        # subject
                        elif test_splitter.split_type == SplitByType.SUBJECT or test_splitter.split_type == SplitByType.SUBJECT_IND:
                            mask, exclude = self.epoch_data.pick_subject(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.SUBJECT_IND:
                                dataset.discard(exclude)
                        idx += 1

                # set result as mask if available
                if not has_next:
                    break
                if idx > 0:
                    dataset.set_test(mask)
                
                # split val data
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
                            mask, exclude = self.epoch_data.pick_session(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        # label
                        elif val_splitter.split_type == ValSplitByType.TRIAL:
                            mask, exclude = self.epoch_data.pick_trial(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        # subject
                        elif val_splitter.split_type == ValSplitByType.SUBJECT:
                            mask, exclude = self.epoch_data.pick_subject(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        idx += 1
                
                # set result as mask if available
                if idx > 0:
                    dataset.set_val(mask)
                dataset.set_train()
                # check job interrupt
                if self.interrupted:
                    raise KeyboardInterrupt
                self.datasets.append(dataset)
                group_idx += 1
                # break at the end if not cross validation
                if not self.config.is_cross_validation:
                    break
            # break at the end if not individual scheme
            if self.config.train_type != TrainingType.IND:
                break
        
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
