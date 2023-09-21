from __future__ import annotations

import numpy as np

from ..utils import validate_type
from .data_splitter import DataSplittingConfig
from .epochs import Epochs


class Dataset:
    """Class for storing splitted dataset.

    Attributes:
        SEQ: int
            Sequence number for generating dataset ID
        name: str
            Name of the dataset
        epoch_data: :class:`Epochs`
            Epoch data to be splitted
        config: :class:`DataSplittingConfig`
            Splitting configuration
        dataset_id: int
            ID of the dataset
        remaining_mask: np.ndarray
            Mask for remaining trials
        train_mask: np.ndarray
            Mask for training set
        val_mask: np.ndarray
            Mask for validation set
        test_mask: np.ndarray
            Mask for test set
        is_selected: bool
            Whether the dataset is selected
    """
    SEQ = 0
    def __init__(self, epoch_data: Epochs, config: DataSplittingConfig):
        validate_type(epoch_data, Epochs, "epoch_data")
        validate_type(config, DataSplittingConfig, "config")
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
    def get_epoch_data(self) -> Epochs:
        """Get the epoch data of the dataset."""
        return self.epoch_data

    def get_name(self) -> str:
        """Get the formatted name of the dataset."""
        return str(self.dataset_id) + '-' + self.name

    def get_ori_name(self) -> str:
        """Get the original name of the dataset."""
        return self.name

    def get_all_trial_numbers(self) -> tuple:
        """Get each number of trials in train, validation and test set.

        Returns:
            (train_number, val_number, test_number)
        """
        train_number = sum(self.train_mask)
        val_number = sum(self.val_mask)
        test_number = sum(self.test_mask)
        return train_number, val_number, test_number

    def get_treeview_row_info(self) -> tuple:
        """Return the information of the dataset for displaying in UI treeview.

        Returns:
            (selected: str,
             name: str,
             train_number: int,
             val_number: int,
             test_number: int
            )
        """
        train_number, val_number, test_number = self.get_all_trial_numbers()
        selected = 'O' if self.is_selected else 'X'
        name = self.get_name()
        return selected, name, train_number, val_number, test_number

    ## setter
    def set_selection(self, select):
        """Set the dataset selection."""
        self.is_selected = select

    def set_name(self, name: str):
        """Set the dataset name."""
        self.name = name

    def has_set_empty(self) -> bool:
        """Return whether the dataset is empty."""
        train_number, val_number, test_number = self.get_all_trial_numbers()
        return train_number == 0 or val_number == 0 or test_number == 0

    ### mask
    def get_remaining_mask(self) -> np.ndarray:
        """Get the mask for remaining trials."""
        return self.remaining_mask.copy()

    ## picker
    def discard_remaining_mask(self, mask: np.ndarray) -> None:
        """Mark all the trials in the mask as discarded."""
        self.remaining_mask &= np.logical_not(mask)

    def set_remaining_by_subject_idx(self, idx: int) -> None:
        """Set the remaining mask to the mask of target subject."""
        self.remaining_mask = self.epoch_data.pick_subject_mask_by_idx(idx)

    ## set result
    def set_test(self, mask: np.ndarray) -> None:
        """Set the mask for test set and update the remaining mask."""
        self.test_mask = mask & self.remaining_mask
        self.remaining_mask &= np.logical_not(mask)

    def set_val(self, mask: np.ndarray) -> None:
        """Set the mask for validation set and update the remaining mask."""
        self.val_mask = mask & self.remaining_mask
        self.remaining_mask &= np.logical_not(mask)

    def set_remaining_to_train(self) -> None:
        """Set the remaining trials as training set."""
        self.train_mask |= self.remaining_mask
        self.remaining_mask &= False

    ## filter
    def intersection_with_subject_by_idx(
        self,
        mask: np.ndarray,
        idx: int
    ) -> np.ndarray:
        """Return the intersection of the mask and the mask of target subject."""
        return mask & self.epoch_data.pick_subject_mask_by_idx(idx)

    # train
    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the training data and label.

        Returns:
            (X, y)
        """
        X = self.epoch_data.get_data()[self.train_mask]
        y = self.epoch_data.get_label_list()[self.train_mask]
        return X, y

    def get_val_data(self) -> tuple:
        """Return the validation data and label.

        Returns:
            (X, y)
        """
        X = self.epoch_data.get_data()[self.val_mask]
        y = self.epoch_data.get_label_list()[self.val_mask]
        return X, y

    def get_test_data(self) -> tuple:
        """Return the test data and label.

        Returns:
            (X, y)
        """
        X = self.epoch_data.get_data()[self.test_mask]
        y = self.epoch_data.get_label_list()[self.test_mask]
        return X, y

    # get data len
    def get_train_len(self) -> int:
        """Return the number of trials in training set."""
        return sum(self.train_mask)

    def get_val_len(self) -> int:
        """Return number of trials in validation set."""
        return sum(self.val_mask)

    def get_test_len(self) -> int:
        """Return number of trials in test set."""
        return sum(self.test_mask)
