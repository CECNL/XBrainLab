from typing import TYPE_CHECKING, List, Optional

from ..utils import validate_list_type, validate_type
from .raw import Raw

if TYPE_CHECKING: # pragma: no cover
    from .. import Study


class RawDataLoader(list):
    """Helper class for loading raw data.

    Validate the loaded raw data consistency and apply to the study.

    Parameters:
        raw_data_list: List of loaded raw data.
    """
    def __init__(self, raw_data_list: Optional[List[Raw]]=None):
        if raw_data_list is None:
            raw_data_list = []
        validate_list_type(raw_data_list, Raw, "raw_data_list")
        super().__init__(raw_data_list)
        if raw_data_list:
            self.validate()

    def get_loaded_raw(self, filepath: str) -> Raw:
        """Return the loaded raw data with the given filepath.

        Args:
            filepath: Filepath of the raw data.
        """
        for raw_data in self:
            if filepath == raw_data.get_filepath():
                return raw_data
        return None

    def validate(self) -> None:
        """Validate the loaded raw data consistency.

        Raises:
            ValueError: If the loaded raw data is inconsistent or empty.
        """
        for i in range(len(self)):
            raw_data = self[i]
            self.check_loaded_data_consistency(raw_data, idx=0)
            _, event_id = raw_data.get_event_list()
            if not event_id:
                raise ValueError(
                    f"No label has been loaded for {raw_data.get_filename()}"
                )
        if len(self) == 0:
            raise ValueError("No dataset has been loaded")

    def check_loaded_data_consistency(self, raw: Raw, idx: int = -1):
        """Validate the loaded raw data consistency with the raw data in the dataset
           at the given index.

        Args:
            raw: Loaded raw data.
            idx: Index of the raw data in the dataset. Default to the last one.

        Raises:
            ValueError: If the loaded raw data is inconsistent with
                        the raw data in the dataset.
        """
        validate_type(raw, Raw, 'raw')
        # valide if the dataset is empty
        if not self:
            return
        # check channel number
        if self[idx].get_nchan() != raw.get_nchan():
            raise ValueError(
                f'Dataset channel numbers inconsistent (got {raw.get_nchan()}).'
            )
        # check sfreq
        if self[idx].get_sfreq() != raw.get_sfreq():
            raise ValueError(
                f'Dataset sample frequency inconsistent (got {raw.get_sfreq()}).'
            )
        # check same data type
        if self[idx].is_raw() != raw.is_raw():
            raise ValueError('Dataset type inconsistent.')
        # check epoch trial size
        if (
            not raw.is_raw() and
            (self[idx].get_epoch_duration() != raw.get_epoch_duration())
        ):
            raise ValueError(
                f'Epoch duration inconsistent (got {raw.get_epoch_duration()}).'
            )

    def append(self, raw: Raw) -> None:
        """Append the loaded raw data to the dataset.

        Args:
            raw: Loaded raw data.
        """
        self.check_loaded_data_consistency(raw)
        super().append(raw)

    def apply(self, study: 'Study') -> None:
        """Apply the loaded raw data to the study.

        Args:
            study: XBrainLab Study to apply the loaded raw data.
        """
        from .. import Study
        validate_type(study, Study, 'study')
        self.validate()
        study.set_loaded_data_list(self)
