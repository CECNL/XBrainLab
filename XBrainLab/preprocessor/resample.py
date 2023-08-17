from .base import PreprocessBase

class Resample(PreprocessBase):

	def get_preprocess_desc(self, sfreq):
		return f"Resample to {sfreq}"

	def _data_preprocess(self, preprocessed_data, sfreq):
		preprocessed_data.get_mne().load_data()
		if preprocessed_data.is_raw():
			new_mne, new_events = preprocessed_data.get_mne().resample(sfreq=sfreq, events=preprocessed_data.raw_events)
			preprocessed_data.set_mne(new_mne)
			preprocessed_data.set_event(new_events, preprocessed_data.raw_event_id)
		else:
			new_mne = preprocessed_data.get_mne().resample(sfreq=sfreq)
			preprocessed_data.set_mne(new_events)