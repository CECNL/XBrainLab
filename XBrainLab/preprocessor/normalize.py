from .base import PreprocessBase
import numpy as np

class Normalize(PreprocessBase):
	"""Preprocessing class for normalizing data.

	Input:
		norm: Normalization method. Can be "zero mean" or "minmax".
	
	"""
	def get_preprocess_desc(self, norm: str):
		return f"{norm} normalization"

	def _data_preprocess(self, preprocessed_data, norm: str):
		preprocessed_data.get_mne().load_data()
		if norm == "zero mean":
			if preprocessed_data.is_raw():
				arrdata =  preprocessed_data.get_mne()._data.copy()
				preprocessed_data.get_mne()._data = arrdata - np.multiply(
					arrdata.mean(axis=-1)[:, None],np.ones_like(arrdata)
				)
			else:
				arrdata =  preprocessed_data.get_mne()._data.copy()
				for ep in range(preprocessed_data.get_epochs_length()):
					arrdata[ep, :, :] =  arrdata[ep, :, :] - np.multiply(
						arrdata[ep, :, :].mean(axis=-1)[:, None],np.ones_like(
							arrdata[ep, :, :]
						)
					)
				preprocessed_data.get_mne()._data = arrdata
		elif norm== "minmax":
			if preprocessed_data.is_raw():
				arrdata =  preprocessed_data.get_mne()._data.copy()
				ch_min, ch_max = (
					np.multiply(
						arrdata.min(axis=-1)[:, None],
						np.ones_like(arrdata)), 
					np.multiply(
						arrdata.max(axis=-1)[:, None],
						np.ones_like(arrdata)
					)
				)
				arrdata = (arrdata-ch_min)/(ch_max-ch_min+1e-12)
				preprocessed_data.get_mne()._data = arrdata
			else:
				arrdata =  preprocessed_data.get_mne()._data.copy()
				for ep in range(preprocessed_data.get_epochs_length()):
					ch_min = np.multiply(
						arrdata[ep, :, :].min(axis=-1)[:, None],
						np.ones_like(arrdata[ep, :, :])
					)
					ch_max = np.multiply(
						arrdata[ep, :, :].max(axis=-1)[:, None],
						np.ones_like(arrdata[ep, :, :])
					)
					arrdata[ep, :, :] = (
						(arrdata[ep, :, :]-ch_min) / 
						(ch_max-ch_min+1e-12)
					)
				preprocessed_data.get_mne()._data = arrdata
