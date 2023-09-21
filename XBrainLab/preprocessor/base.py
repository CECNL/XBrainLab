from ..load_data import Raw
from ..utils import validate_list_type
from copy import deepcopy
from typing import List

class PreprocessBase:
    """Base class for preprocessors.
    
    Attributes:
        preprocessed_data_list: List[:class:`XBrainLab.preprocessor.Raw`]
            List of preprocessed data.
    """
    def __init__(self, preprocessed_data_list: List[Raw]):
        self.preprocessed_data_list = deepcopy(preprocessed_data_list)
        self.check_data()

    def check_data(self) -> None:
        """Check if the data is valid.

        Raises:
            ValueError: If the data is either empty, 
                        not a list of :class:`XBrainLab.preprocessor.Raw` or not valid.
        """
        if not self.preprocessed_data_list:
            raise ValueError("No valid data is loaded")
        validate_list_type(self.preprocessed_data_list, Raw, 'preprocessed_data_list')
        
    def get_preprocessed_data_list(self) -> List[Raw]:
        """Get the preprocessed data list."""
        return self.preprocessed_data_list

    def get_preprocess_desc(self, *args, **kargs) -> str:
        """Return description of the preprocess."""
        raise NotImplementedError
    
    def data_preprocess(self, *args, **kargs) -> List[Raw]:
        """Wrapper for :meth:`_data_preprocess`."""
        for preprocessed_data in self.preprocessed_data_list:
            self._data_preprocess(preprocessed_data, *args, **kargs)
            preprocessed_data.add_preprocess(self.get_preprocess_desc(*args, **kargs))
        return self.preprocessed_data_list

    def _data_preprocess(self, preprocessed_data: Raw, *args, **kargs) -> None:
        """Preprocess the data."""
        raise NotImplementedError