from .base import PreprocessBase

class Filtering(PreprocessBase):

    def get_preprocess_desc(self, l_freq, h_freq):
        return f"Filtering {l_freq} ~ {h_freq}"

    def _data_preprocess(self, preprocessed_data, l_freq, h_freq):
        preprocessed_data.get_mne().load_data()
        new_mne = preprocessed_data.get_mne().filter(l_freq=l_freq, h_freq=h_freq)
        preprocessed_data.set_mne(new_mne)
