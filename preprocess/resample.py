from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
from copy import deepcopy
import tkinter as tk

class Resample(TopWindow):
	command_label = "Resample"
	def __init__(self, parent, preprocessed_data):
		super().__init__(parent, "Resample")
		self.preprocessed_data = preprocessed_data
		self.check_data()

		self.return_data = None
		self.mne_data = deepcopy(self.preprocessed_data.mne_data)
		data_field = ["sfreq"]
		self.field_var = {key: tk.StringVar() for key in data_field}

		tk.Label(self, text="Sampling Rate: ").grid(row=6, column=0, sticky="w")
		tk.Entry(self, textvariable=self.field_var['sfreq'], bg="White").grid(row=6, column=1, sticky="w")
		tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

	def check_data(self):
		if not (type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
			raise InitWindowValidateException(window=self, message="No valid data is loaded")

	def _data_preprocess(self):
		# Check Input is Valid
		if self.field_var['sfreq'].get() == "":
			raise InitWindowValidateException(window=self, message="No Input")
		elif float(self.field_var['sfreq'].get()) < 0.0:
			raise InitWindowValidateException(window=self, message="Input value invalid")

		if type(self.preprocessed_data) == Raw:
			new_events = {}
			for fn, mne_data in self.mne_data.items():
				self.mne_data[fn], new_events[fn] = mne_data.resample(sfreq=float(self.field_var['sfreq'].get()), events=self.preprocessed_data.raw_events[fn])
			self.return_data = Raw(self.preprocessed_data.raw_attr, self.mne_data, new_events, self.preprocessed_data.event_id)
		else:
			for fn, mne_data in self.mne_data.items():
				self.mne_data[fn] = mne_data.resample(sfreq=float(self.field_var['sfreq'].get()))
			self.return_data = Epochs(self.preprocessed_data.epoch_attr, self.mne_data)
		self.destroy()

	def _get_result(self):
		return self.return_data
