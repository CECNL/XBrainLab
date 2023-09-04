from .base import PreprocessBase
from ..load_data import Raw

class Filtering(PreprocessBase):
    """Preprocessing class for filtering data.

    Input:
        l_freq: Low frequency.
        h_freq: High frequency.
    """

    def get_preprocess_desc(self, l_freq: float, h_freq: float):
        return f"Filtering {l_freq} ~ {h_freq}"

    def _data_preprocess(self, preprocessed_data: Raw, l_freq: float, h_freq: float):
        preprocessed_data.get_mne().load_data()
        new_mne = preprocessed_data.get_mne().filter(l_freq=l_freq, h_freq=h_freq)
        preprocessed_data.set_mne(new_mne)
