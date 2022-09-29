from .base import PreprocessBase
from ..base import TopWindow, ValidateException, InitWindowValidateException
import tkinter as tk
import mne

class WindowEpoch(PreprocessBase):
    command_label = "Window Epoch"
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Window Epoch", preprocessed_data_list)
        
        data_field = ["duration", "overlap", "baseline_tmin", "baseline_tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Duration of each epoch (sec): ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['duration']).grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Overlap between epochs (sec): ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['overlap']).grid(row=3, column=1, sticky="w")
        tk.Label(self, text="").grid(row=4, column=0, sticky="w")

        tk.Button(self, text="Confirm", command=self._extract_epoch, width=8).grid(row=9, columnspan=2)

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if not preprocessed_data.is_raw():
                raise InitWindowValidateException(window=self, message=f"Only raw data can be epoched, got epochs")
            events, event_id = preprocessed_data.get_event_list()
            if not event_id:
                raise InitWindowValidateException(window=self, message=f"No event markers found for {preprocessed_data.get_filename()}")
            if len(events) != 1 or len(event_id) != 1:
                raise InitWindowValidateException(window=self, message=f"Should only contain single event label, found events={len(events)}, event_id={len(event_id)}")

    def get_preprocess_desc(self, duration, overlap):
        return f"Epoching {duration}s ({overlap}s overlap) by sliding window"

    def _extract_epoch(self):
        if not self.field_var['duration'].get().strip():
            raise ValidateException(window=self, message="No duration input")

        for preprocessed_data in self.preprocessed_data_list:
            mne_data = preprocessed_data.get_mne()
            duration = float(self.field_var['duration'].get())
            overlap = 0.0 if self.field_var['overlap'].get() == "" else float(self.field_var['overlap'].get())
            FIXED_ID = 0
            epoch = mne.make_fixed_length_epochs(mne_data, duration=duration, overlap=overlap, preload=True, id=FIXED_ID)
            _, event_id = preprocessed_data.get_raw_event_list()
            epoch.event_id[event_id.keys()[0]] = FIXED_ID
            try:
                preprocessed_data.set_mne(epoch)
                preprocessed_data.add_preprocess(self.get_preprocess_desc(duration, overlap))
            except:
                raise ValidateException(self, f'Inconsistent number of events with label length (got {len(events)})')

        self.return_data = self.preprocessed_data_list
        self.destroy()
