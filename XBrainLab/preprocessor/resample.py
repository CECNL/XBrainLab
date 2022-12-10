from .base import PreprocessBase

class Resample(PreprocessBase):

	def get_preprocess_desc(self, sfreq):
		return f"Resample to {sfreq}"

	def _data_preprocess(self, preprocessed_data, sfreq):
		preprocessed_data.get_mne().load_data()
		preprocessed_data.get_mne().resample(sfreq=sfreq)