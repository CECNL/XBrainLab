import tkinter as tk
import mne

from ..base import TopWindow, ValidateException, InitWindowValidateException
from .base import PreprocessBase

from XBrainLab import preprocessor as Preprocessor

class TimeEpoch(PreprocessBase):
    command_label = "Time Epoch"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.TimeEpoch(preprocessed_data_list)
        super().__init__(parent, "Time Epoch", preprocessor)

        data_field = ["select_events", "epoch_tmin", "epoch_tmax", "baseline_tmin", "baseline_tmax", "doRemoval"]
        self.field_var = {key: tk.StringVar() for key in data_field}
        self.event_id = None
        self.return_data = None

        self.rowconfigure(list(range(6)), weight=1)
        self.columnconfigure([1], weight=1)

        tk.Label(self, text="Choose Events: ").grid(row=0, column=0, sticky="w", padx=5)
        tk.Label(self, textvariable=self.field_var['select_events']).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        tk.Button(self, text="...", command=self._choose_events, width=8).grid(row=0, column=2, padx=5, pady=5)
        tk.Label(self, text="Epoch limit start: ").grid(row=1, column=0, sticky="w", padx=5)
        tk.Entry(self, textvariable=self.field_var['epoch_tmin']).grid(row=1, column=1, sticky="ew", padx=5)
        tk.Label(self, text="Epoch limit end: ").grid(row=2, column=0, sticky="w", padx=5)
        tk.Entry(self, textvariable=self.field_var['epoch_tmax']).grid(row=2, column=1, sticky="ew", padx=5)

        tk.Checkbutton(self, text='Do Baseline Removal', variable=self.field_var['doRemoval'], onvalue=1, offvalue=0, command=self._click_checkbox).grid(row=3, columnspan=3, sticky="w", pady=(15, 2))
        tk.Label(self, text="min: ").grid(row=4, column=0, sticky="w", padx=5)
        self.min_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmin'])
        self.min_entry.grid(row=4, column=1, sticky="ew", padx=5)
        tk.Label(self, text="max: ").grid(row=5, column=0, sticky="w", padx=5)
        self.max_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmax'])
        self.max_entry.grid(row=5, column=1, sticky="ew", padx=5)
        self.field_var['doRemoval'].set(1)
        tk.Button(self, text="Confirm", command=self._extract_epoch, width=8).grid(row=6, columnspan=3)

    def _click_checkbox(self):
        if self.field_var['doRemoval'].get() == "1":
            self.min_entry.config(state="normal")
            self.max_entry.config(state="normal")
        else:
            self.min_entry.config(state="disabled")
            self.max_entry.config(state="disabled")

    def _choose_events(self):
        event_id = SelectEvents(self, self.preprocessor.get_preprocessed_data_list()).get_result()
        if event_id:
            self.event_id = event_id
            self.field_var['select_events'].set(",".join([str(e) for e in event_id]))

    def _extract_epoch(self):
        baseline = None
        if self.field_var['doRemoval'].get() == "1":
            baseline_tmin = float(self.field_var['baseline_tmin'].get()) if self.field_var['baseline_tmin'].get() != "" else None
            baseline_tmax = float(self.field_var['baseline_tmax'].get()) if self.field_var['baseline_tmax'].get() != "" else None
            baseline = (baseline_tmin, baseline_tmax)

        # Check input value
        if self.field_var['epoch_tmin'].get() == "" or self.field_var['epoch_tmax'].get() == "":
            raise ValidateException(window=self, message="Invalid epoch range")

        if not self.event_id:
            raise ValidateException(window=self, message="No event was selected")
        selected_event_names = list(self.event_id.keys())
        
        try:
            tmin = float(self.field_var['epoch_tmin'].get())
            tmax = float(self.field_var['epoch_tmax'].get())
            self.return_data = self.preprocessor.data_preprocess(baseline, selected_event_names, tmin, tmax)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        
        self.script_history.add_cmd(f'selected_event_names={repr(selected_event_names)}')
        self.script_history.add_cmd(f'baseline={repr(baseline)}')
        self.script_history.add_cmd(f'tmin={repr(tmin)}')
        self.script_history.add_cmd(f'tmax={repr(tmax)}')
        self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.TimeEpoch, baseline=baseline, selected_event_names=selected_event_names, tmin=tmin, tmax=tmax)')
        self.ret_script_history = self.script_history

        self.destroy()

class SelectEvents(TopWindow):
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Select Events")
        self.preprocessed_data_list = preprocessed_data_list
        self.event_id = None

        self.rowconfigure([1], weight=1)
        self.columnconfigure([0], weight=1)

        event_id_set = set()
        for preprocessed_data in preprocessed_data_list:
            _, event_id = preprocessed_data.get_raw_event_list()
            if len(event_id) == 0:
                _, event_id = preprocessed_data.get_event_list()
            event_id_set.update(event_id)
        event_id_set = list(event_id_set)

        scrollbar = tk.Scrollbar(self)
        self.listbox = tk.Listbox(self, selectmode="multiple", yscrollcommand=scrollbar.set)        
        for event_name in event_id_set:
            self.listbox.insert(tk.END, event_name)
        scrollbar.config(command=self.listbox.yview)
        
        tk.Label(self, text="Choose Events: ").grid(row=0, column=0, columnspan=2)
        self.listbox.grid(row=1, column=0, padx=10, pady=10, sticky='news')
        scrollbar.grid(row=1, column=1, pady=10, sticky='news')
        tk.Button(self, text="Confirm", command=self._getEventID, width=8).grid(row=2, column=0, columnspan=2)
        
        self.event_id_set = event_id_set

    def _getEventID(self):
        event_id = {}
        listbox_idx = list(self.listbox.curselection())
        if len(listbox_idx) == 0:
            raise ValidateException(window=self, message="No event was selected")
        for idx in listbox_idx:
            event_id[self.event_id_set[idx]] = len(event_id)
        self.event_id = event_id
        self.destroy()

    def _get_result(self):
        return self.event_id
