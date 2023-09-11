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
			events, event_id = preprocessed_data.get_event_list()
			if len(events) == 0:
				new_mne = preprocessed_data.get_mne().resample(sfreq=sfreq)
				preprocessed_data.set_mne(new_mne)
			else:
				new_mne, new_events = preprocessed_data.get_mne().resample(sfreq=sfreq, events=events)
				preprocessed_data.set_mne(new_mne)
				preprocessed_data.set_event(new_events, event_id)
		else:
			new_mne = preprocessed_data.get_mne().resample(sfreq=sfreq)
			preprocessed_data.set_mne_and_wipe_events(new_mne)
