from .base import PreprocessBase
from ..base import ValidateException
import tkinter as tk

class Resample(PreprocessBase):
	command_label = "Resample"
	def __init__(self, parent, preprocessed_data_list):
		super().__init__(parent, "Resample", preprocessed_data_list)
		data_field = ["sfreq"]
		self.field_var = {key: tk.StringVar() for key in data_field}

		self.rowconfigure([0,1], weight=1)
		self.columnconfigure([0,1], weight=1)

		tk.Label(self, text="Sampling Rate: ").grid(row=0, column=0, pady=10, padx=5)
		tk.Entry(self, textvariable=self.field_var['sfreq']).grid(row=0, column=1, pady=10, padx=5)
		tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=1, columnspan=2)

	def get_preprocess_desc(self, sfreq):
		return f"Resample to {sfreq}"

	def _data_preprocess(self):
		# Check Input is Valid
		if self.field_var['sfreq'].get().strip() == "":
			raise ValidateException(window=self, message="No Input")
		
		for preprocessed_data in self.preprocessed_data_list:
			preprocessed_data.get_mne().load_data()
			preprocessed_data.get_mne().resample(sfreq=float(self.field_var['sfreq'].get()))
			preprocessed_data.add_preprocess(self.get_preprocess_desc(self.field_var['sfreq'].get()))
			
		self.return_data = self.preprocessed_data_list
		self.destroy()