from typing import List
import numpy as np

from .option import TrainingType, SplitByType, ValSplitByType
from .dataset import Dataset
from . import DataSplittingConfig, Epochs

from ..utils import validate_type

class DatasetGenerator:
    """Class for generating dataset from epoch data and splitting configuration.
    
    Attributes:
        epoch_data: :class:`Epochs`
            Epoch data to be splitted
        config: :class:`DataSplittingConfig`
            Splitting configuration
        datasets: list[:class:`Dataset`]
            List of generated datasets
        interrupted: bool
            Whether the dataset generation is interrupted
        preview_failed: bool
            Whether the preview failed
        test_splitter_list: List[`DataSplitter`]
            List of splitters for test set
        val_splitter_list: List[`DataSplitter`]
            List of splitters for validation set
    """
    def __init__(
        self, 
        epoch_data: Epochs, 
        config: DataSplittingConfig, 
        datasets: List[Dataset] = None
    ):
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
        self.done = False

    """
    How it works:
        (Enter generate)
        Call the handle function for given train_type
        (Enter handle_XXX)
        If the train_type (XXX) is IND
            For each subject
                Set name with subject as prefix
                Call the handle function
        else if the train_type (XXX) is FULL
            Set name with 'Group' as prefix
            Call the handle function
        (Enter handle)
        While remaining epoch is none (first loop) or not empty (cross validation)
            Create a new dataset configured with the given epoch_data and config
            If the train_type is IND
                Filter out target subject
            set clean_mask with whole available epochs
            set mask to remaining_mask if remaining_mask is not none (cross validation)
            (this is to skip previous selected epochs as test set to 
                make sure all test set are different)
            split test set and get the remaining_mask for next cross validation
            split validation set
            set remaining epochs to train set
            append the dataset to datasets
    
    How cross validation works:
        For each fold
            If it is the first fold
                Set mask to whole available epochs
            else
                Set mask to remaining_mask that excluded all previous selected epochs
                    that are set as test set
            (Enter split_test with mask)
            For each test_splitter
                split by condition
                If it is the first splitter
                    Keep the remaining_mask for next cross validation

    How independent works:
        In split_test, the remaining_mask is set to False 
        to discard all epochs that are dependent to the current test set
    """
    def handle_IND(self) -> None:
        """Wrapper for generating datasets for individual scheme.
           Called by :func:`generate`."""
        for subject_idx in range(len(self.epoch_data.get_subject_index_list())):
            name_prefix = f"Subject-{self.epoch_data.get_subject_name(subject_idx)}"
            def hook(dataset):
                dataset.set_remaining_by_subject_idx(subject_idx)
            self.handle(name_prefix, hook)
    
    def handle_FULL(self) -> None:
        """Wrapper for generating datasets for full scheme. 
           Called by :func:`generate`."""
        name_prefix = "Group"
        self.handle(name_prefix)
    
    def split_test(
        self, 
        dataset: Dataset, 
        group_idx: int, 
        mask: np.ndarray, 
        clean_mask: np.ndarray
    ) -> np.ndarray:
        """Split the test set of the dataset.

        Args:
            dataset: Dataset to be splitted
            group_idx: Index of the group
            mask: Mask to filter out remaining epochs, 
                  ecxluding already selected cross validation part. 
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs, 
                        including all available selection. 1D np.ndarray of bool.
        
        Returns:
            Mask to filter out remaining epochs, 
            ecxluding already selected cross validation part. 
            1D np.ndarray of bool.
        """
        idx = 0
        next_mask = mask.copy()
        for test_splitter in self.test_splitter_list:
            if test_splitter.is_option:
                if self.interrupted:
                    self.preview_failed = True
                    raise KeyboardInterrupt
                if not test_splitter.is_valid():
                    self.preview_failed = True
                    raise ValueError("Preview failed")
                # session
                if (
                    test_splitter.split_type == SplitByType.SESSION or 
                    test_splitter.split_type == SplitByType.SESSION_IND
                ):
                    split_func = self.epoch_data.pick_session
                # label
                elif (
                    test_splitter.split_type == SplitByType.TRIAL or 
                    test_splitter.split_type == SplitByType.TRIAL_IND
                ):
                    split_func = self.epoch_data.pick_trial
                # subject
                elif (
                    test_splitter.split_type == SplitByType.SUBJECT or 
                    test_splitter.split_type == SplitByType.SUBJECT_IND
                ):
                    split_func = self.epoch_data.pick_subject
                else:
                    raise NotImplementedError
                mask, excluded = split_func(
                    mask=mask, clean_mask=clean_mask, 
                    value=test_splitter.get_value(), 
                    split_unit=test_splitter.get_split_unit(), 
                    group_idx=group_idx
                )
                # save for next cross validation
                if idx == 0:
                    next_mask = excluded.copy()
                    # restore previous cross validation part
                    excluded = clean_mask & np.logical_not(mask)
                    clean_mask = None
                if not mask.any():
                    break
                # independent
                if (
                    test_splitter.split_type == SplitByType.SESSION_IND or 
                    test_splitter.split_type == SplitByType.TRIAL_IND or 
                    test_splitter.split_type == SplitByType.SUBJECT_IND
                ):
                    dataset.discard_remaining_mask(excluded)
                idx += 1
        if idx > 0:
            dataset.set_test(mask)
        else:
            next_mask &= False
        return next_mask

    def split_validate(self, dataset: Dataset, group_idx: int) -> None:
        """Split the validation set of the dataset.
        
        Args:
            dataset: Dataset to be splitted
            group_idx: Index of the group
        """
        mask = dataset.get_remaining_mask()
        idx = 0
        for val_splitter in self.val_splitter_list:
            if val_splitter.is_option:
                # check job interrupt
                if self.interrupted:
                    raise KeyboardInterrupt
                if not val_splitter.is_valid():
                    self.preview_failed = True
                    raise ValueError("Preview failed")
                # session
                if val_splitter.split_type == ValSplitByType.SESSION:
                    split_func = self.epoch_data.pick_session
                # label
                elif val_splitter.split_type == ValSplitByType.TRIAL:
                    split_func = self.epoch_data.pick_trial
                # subject
                elif val_splitter.split_type == ValSplitByType.SUBJECT:
                    split_func = self.epoch_data.pick_subject
                else:
                    raise NotImplementedError
                mask, _ = split_func(
                    mask=mask, clean_mask=None, 
                    value=val_splitter.get_value(), 
                    split_unit=val_splitter.get_split_unit(),
                    group_idx=group_idx
                )
                idx += 1
        if idx > 0:
            dataset.set_val(mask)

    def handle(self, name_prefix: str, dataset_hook: callable = None) -> None:
        """Internal function for generating datasets
        
        Args:
            name_prefix: Prefix of dataset name
            dataset_hook: Function for setting up dataset for specific scheme
        """
        group_idx = 0
        remaining_mask = None
        while (
            (remaining_mask is None) or 
            (self.config.is_cross_validation and remaining_mask.any())
        ):
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
            self.split_validate(dataset, group_idx)
            dataset.set_remaining_to_train()
            
            self.datasets.append(dataset)
            group_idx += 1

    def generate(self) -> List[Dataset]:
        """Internal function for calling the dataset generation function."""
        if not self.is_clean():
            raise ValueError(
                'Dataset generation is not clean. Reset the generator and try again.'
            )
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

    def set_interrupt(self) -> None:
        """Set the interrupt flag to break the dataset generation."""
        self.preview_failed = True
        self.interrupted = True

    def prepare_reuslt(self) -> list:
        """Generate the datasets and remove unselcted datasets."""
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
        self.done = True
        return self.datasets

    def is_clean(self) -> bool:
        """Check if the dataset generation is clean."""
        return self.done or (not self.interrupted and not self.preview_failed)
    
    def reset(self) -> None:
        """Reset the dataset generator."""
        self.datasets = None
        self.interrupted = False
        self.preview_failed = False
        Dataset.SEQ = 0
    
    def apply(self, study) -> None:
        """Apply the generated datasets to the study."""
        from ..lab import XBrainLab
        validate_type(study, XBrainLab, 'study')
        self.prepare_reuslt()
        study.set_datasets(self.datasets)
