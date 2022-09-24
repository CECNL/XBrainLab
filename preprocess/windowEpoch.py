from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
from copy import deepcopy
import mne

class WindowEpoch(TopWindow):
	command_label = "Window Epoch"
	def __init__(self, parent, preprocessed_data):
		super().__init__(parent, "Window Epoch")
		self.preprocessed_data = preprocessed_data.copy()
		self.check_data()

		self.return_data = None
		data_field = ["duration", "overlap", "baseline_tmin", "baseline_tmax", "doRemoval"]
		self.field_var = {key: tk.StringVar() for key in data_field}

		tk.Label(self, text="Duration of each epoch: ").grid(row=2, column=0, sticky="w")
		tk.Entry(self, textvariable=self.field_var['duration'], bg="White").grid(row=2, column=1, sticky="w")
		tk.Label(self, text="Overlap between epochs: ").grid(row=3, column=0, sticky="w")
		tk.Entry(self, textvariable=self.field_var['overlap'], bg="White").grid(row=3, column=1, sticky="w")
		tk.Label(self, text="").grid(row=4, column=0, sticky="w")

		tk.Checkbutton(self, text='Do Baseline Removal', variable=self.field_var['doRemoval'], onvalue=1, offvalue=0, command=lambda win=self: self._click_checkbox()).grid(row=5, columnspan=2, sticky="w")
		tk.Label(self, text="latency range").grid(row=6, column=0, sticky="w")
		tk.Label(self, text="min: ").grid(row=7, column=0, sticky="w")
		self.min_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmin'], bg="White")
		self.min_entry.grid(row=7, column=1, sticky="w")
		tk.Label(self, text="max: ").grid(row=8, column=0, sticky="w")
		self.max_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmax'], bg="White")
		self.max_entry.grid(row=8, column=1, sticky="w")
		self.field_var['doRemoval'].set(0)
		self.min_entry.config(state="disabled")
		self.max_entry.config(state="disabled")
		tk.Button(self, text="Confirm", command=lambda win=self: self._extract_epoch(), width=8).grid(row=9, columnspan=2)

	def check_data(self):
		if type(self.preprocessed_data) != Raw:
			raise InitWindowValidateException(window=self, message="No valid data is loaded")

	def _click_checkbox(self):
		if self.field_var['doRemoval'].get() == "1":
			self.min_entry.config(state="normal")
			self.max_entry.config(state="normal")
		else:
			self.min_entry.config(state="disabled")
			self.max_entry.config(state="disabled")

	def _extract_epoch(self):
		if self.field_var['duration'].get() == "":
			raise InitWindowValidateException(window=self, message="No Input")

		self.data_list = {}
		for fn, mne_data in self.preprocessed_data.mne_data.items():
			overlap = 0.0 if self.field_var['overlap'].get() == "" else float(self.field_var['overlap'].get())
			epoch = mne.make_fixed_length_epochs(mne_data, duration=float(self.field_var['duration'].get()), overlap=overlap, preload=True)
			if self.field_var['doRemoval'].get() == "1":
				baseline_tmin = float(self.field_var['baseline_tmin'].get()) if self.field_var['baseline_tmin'].get() != "" else None
				baseline_tmax = float(self.field_var['baseline_tmax'].get()) if self.field_var['baseline_tmax'].get() != "" else None
				epoch.average().apply_baseline((baseline_tmin, baseline_tmax))
			self.data_list[fn] = epoch

		self.return_data = Epochs(self.preprocessed_data.raw_attr, self.data_list)
		self.destroy()

	def _get_result(self):
		return self.return_data
