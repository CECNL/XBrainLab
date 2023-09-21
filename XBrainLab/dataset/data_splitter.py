from __future__ import annotations

from ..utils import validate_list_type, validate_type
from .option import SplitByType, SplitUnit, TrainingType, ValSplitByType


class DataSplitter:
    """Class for data splitting configuration.

    Attributes:
        split_type: :class:`SplitByType`
            Type of splitting action
        value_var: str | None
            String representation of splitting value.
        split_unit: :class:`SplitUnit` | None
            Unit of splitting value
        is_option: bool
            Whether this splitter is real option or just a label
        text: str
            String representation of :attr:`split_type`
    """
    def __init__(
        self,
        split_type: SplitByType | ValSplitByType,
        value_var: str | None = None,
        split_unit: SplitUnit | None = None,
        is_option: bool = True
    ):
        validate_type(split_type, (SplitByType, ValSplitByType), "split_type")
        if split_unit:
            validate_type(split_unit, SplitUnit, "split_unit")
        self.is_option = is_option
        self.split_type = split_type
        self.text = split_type.value
        self.value_var = value_var
        self.split_unit = split_unit

    def is_valid(self) -> bool:
        """Check whether the value matches the unit and all constraints."""

        # check if all required fields are filled
        if self.value_var is None:
            return False
        if self.split_unit is None:
            return False

        # ratio: should be float
        if self.split_unit == SplitUnit.RATIO:
            try:
                val = float(self.value_var)
                if 0 <= val <= 1:
                    return True
            except ValueError:
                return False
        # number: should be int
        elif self.split_unit == SplitUnit.NUMBER:
            return str(self.value_var).isdigit()
        # kfold: should be int > 0
        elif self.split_unit == SplitUnit.KFOLD:
            val = str(self.value_var)
            if val.isdigit():
                return int(val) > 0
        # manual: should be list of int separated by space
        elif self.split_unit == SplitUnit.MANUAL:
            val = str(self.value_var).strip()
            vals = val.split(' ')
            return all(not (len(val.strip()) > 0 and not val.isdigit()) for val in vals)
        else:
            raise NotImplementedError

        return False

    # getter
    def get_value(self) -> float | list[int]:
        """Get option value based on split unit.

        Returns:
            List[int]: if :attr:`split_unit` is :attr:`SplitUnit.MANUAL`
            float: otherwise
        """
        if not self.is_valid():
            raise ValueError("Splitter is not valid")
        if self.split_unit == SplitUnit.MANUAL:
            return [
                int(i)
                for i in self.value_var.strip().split(' ')
                if len(i.strip()) > 0
            ]
        else:
            return float(self.value_var)

    def get_raw_value(self) -> str:
        """Get :attr:`value_var`."""
        if not self.is_valid():
            raise ValueError("Splitter is not valid")
        return self.value_var

    def get_split_unit(self) -> SplitUnit:
        """Get :attr:`split_unit`."""
        return self.split_unit

    def get_split_unit_repr(self) -> str:
        """Get string representation of :attr:`split_unit`."""
        return f"{self.split_unit.__class__.__name__}.{self.split_unit.name}"

    def get_split_type_repr(self) -> str:
        """Get string representation of :attr:`split_type`."""
        return f"{self.split_type.__class__.__name__}.{self.split_type.name}"

class DataSplittingConfig:
    """Utility class for storing data splitting configuration for a training scheme.

    Attributes:
        train_type: :class:`TrainingType`
            TrainingType
        is_cross_validation: bool
            Whether to use cross validation
        val_splitter_list: List[:class:`DataSplitter`]
            list of DataSplitter for validation set
        test_splitter_list: List[:class:`DataSplitter`]
            list of DataSplitter for test set
    """
    def __init__(
        self,
        train_type: TrainingType,
        is_cross_validation: bool,
        val_splitter_list: list[DataSplitter],
        test_splitter_list: list[DataSplitter]
    ):
        validate_type(train_type, TrainingType, "train_type")
        validate_type(is_cross_validation, bool, "is_cross_validation")
        validate_list_type(val_splitter_list, DataSplitter, "val_splitter_list")
        validate_list_type(test_splitter_list, DataSplitter, "test_splitter_list")

        self.train_type = train_type # TrainingType
        self.is_cross_validation = is_cross_validation
        self.val_splitter_list = val_splitter_list
        self.test_splitter_list = test_splitter_list

    def get_splitter_option(self) -> (list[DataSplitter], list[DataSplitter]):
        """Get list of DataSplitter for validation set and test set."""
        return self.val_splitter_list, self.test_splitter_list

    def get_train_type_repr(self) -> str:
        """Get string representation of :attr:`train_type`."""
        return f"{self.train_type.__class__.__name__}.{self.train_type.name}"

