from .base import PreprocessBase
from ..load_data import Raw
from typing import List
class Resample(PreprocessBase):
	"""Preprocessing class for resampling data.

	Input:
		sfreq: Sampling frequency.
	"""

	def get_preprocess_desc(self, sfreq: float):
		return f"Resample to {sfreq}"

	def _data_preprocess(self, preprocessed_data: Raw, sfreq: float):
		preprocessed_data.get_mne().load_data()
		if preprocessed_data.is_raw():
			new_mne, new_events = preprocessed_data.get_mne().resample(sfreq=sfreq, events=preprocessed_data.raw_events)
			preprocessed_data.set_mne(new_mne)
			preprocessed_data.set_event(new_events, preprocessed_data.raw_event_id)
		else:
			new_mne = preprocessed_data.get_mne().resample(sfreq=sfreq)
			preprocessed_data.set_mne(new_events)