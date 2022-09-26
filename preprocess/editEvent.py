from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk

class EditEvent(TopWindow):
	#  menu state disable
	command_label = "Edit Event"
	def __init__(self, parent, preprocessed_data):
		super().__init__(parent, "Edit Event")
		self.old_event = preprocessed_data.event_id
		self.preprocessed_data = preprocessed_data
		self.new_event_name = {k: tk.StringVar() for k in self.old_event.keys()}
		self.new_event = {}
		self.check_data()

		eventidframe = tk.LabelFrame(self, text="Event ids:")
		eventidframe.grid(row=0, column=0, columnspan=2, sticky='w')
		i = 0
		for k, v in self.old_event.items():
			self.new_event_name[k].set(k)
			tk.Entry(eventidframe, textvariable=self.new_event_name[k], width=10).grid(row=i, column=0)
			tk.Label(eventidframe, text=str(v), width=10).grid(row=i, column=1)
			i += 1

		tk.Button(self, text="Cancel", command=lambda: self._confirm(0)).grid(row=1, column=0)
		tk.Button(self, text="Confirm", command=lambda: self._confirm(1)).grid(row=1, column=1)

	def check_data(self):
		if not any([isinstance(self.preprocessed_data, Raw), isinstance(self.preprocessed_data, Epochs)]):
			raise InitWindowValidateException(self, 'No valid data were loaded.')
		if self.preprocessed_data.event_id == {}:
			raise InitWindowValidateException(self, 'Lacking events in loaded data.')

	def _confirm(self, confirm_bool=0):
		if confirm_bool == 1:
			# get from entry to new_event dict
			for i in range(len(self.old_event)):
				if len(set([v.get() for v in self.new_event_name.values()])) < len(self.old_event):
					raise ValidateException(window=self, message="Duplicate event name.")

			# update parent event data
			for k in self.old_event.keys():
				self.new_event[self.new_event_name[k].get()] = self.old_event[k]
			self.preprocessed_data.event_id = self.new_event

			if isinstance(self.preprocessed_data, Raw):  # Raw
				self.preprocessed_data.event_id = self.new_event
			else:
				for mne_struct in self.preprocessed_data.mne_data.values():
					mne_struct.event_id = self.new_event
				self.preprocessed_data.fix_event_id()
		self.destroy()

	def _get_result(self):
		return self.preprocessed_data