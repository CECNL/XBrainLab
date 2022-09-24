from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
from copy import deepcopy
import mne

class TimeEpoch(TopWindow):
    command_label = "Time Epoch"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Time Epoch")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        data_field = ["select_events", "epoch_tmin", "epoch_tmax", "baseline_tmin", "baseline_tmax", "doRemoval"]
        self.field_var = {key: tk.StringVar() for key in data_field}
        self.baseline = None
        self.new_data = None
        self.return_data = None

        tk.Label(self, text="Choose Events: ").grid(row=1, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['select_events'], bg="White").grid(row=1, column=1, sticky="w")
        tk.Button(self, text="...", command=lambda win=self: self._choose_events(), width=8).grid(row=1, column=2)
        tk.Label(self, text="Epoch limit start: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['epoch_tmin'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Epoch limit end: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['epoch_tmax'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Label(self, text="").grid(row=4, columnspan=2, sticky="w")

        tk.Checkbutton(self, text='Do Baseline Removal', variable=self.field_var['doRemoval'], onvalue=1, offvalue=0, command=lambda win=self: self._click_checkbox()).grid(row=5, columnspan=2, sticky="w")
        tk.Label(self, text="latency range").grid(row=6, column=0, sticky="w")
        tk.Label(self, text="min: ").grid(row=7, column=0, sticky="w")
        self.min_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmin'], bg="White")
        self.min_entry.grid(row=7, column=1, sticky="w")
        tk.Label(self, text="max: ").grid(row=8, column=0, sticky="w")
        self.max_entry = tk.Entry(self, textvariable=self.field_var['baseline_tmax'], bg="White")
        self.max_entry.grid(row=8, column=1, sticky="w")
        self.field_var['doRemoval'].set(1)
        tk.Button(self, text="Confirm", command=lambda win=self: self._extract_epoch(), width=8).grid(row=9, columnspan=3)

    def check_data(self):
        if type(self.preprocessed_data) != Raw:
            raise InitWindowValidateException(window=self, message="Invalid data")

    def _click_checkbox(self):
        if self.field_var['doRemoval'].get() == "1":
            self.min_entry.config(state="normal")
            self.max_entry.config(state="normal")
        else:
            self.min_entry.config(state="disabled")
            self.max_entry.config(state="disabled")

    def _choose_events(self):
        self.new_data = SelectEvents(self, self.preprocessed_data).get_result()

        # Check if event is selected
        if self.new_data is None:
            raise ValidateException(window=self, message="No Event is Selected")

        self.field_var['select_events'].set(",".join(list(self.new_data.event_id)))

    def _extract_epoch(self):
        if self.field_var['doRemoval'].get() == "1":
            baseline_tmin = float(self.field_var['baseline_tmin'].get()) if self.field_var['baseline_tmin'].get() != "" else None
            baseline_tmax = float(self.field_var['baseline_tmax'].get()) if self.field_var['baseline_tmax'].get() != "" else None
            self.baseline = (baseline_tmin, baseline_tmax)

        # Check input value
        if self.field_var['epoch_tmin'].get() == "" or self.field_var['epoch_tmax'].get() == "":
            raise InitWindowValidateException(window=self, message="Invalid Value")

        self.data_list = {}
        for fn, mne_data in self.new_data.mne_data.items():
            self.data_list[fn] = mne.Epochs(self.new_data.mne_data[fn], self.new_data.raw_events[fn],
                                            tmin=float(self.field_var['epoch_tmin'].get()),
                                            tmax=float(self.field_var['epoch_tmax'].get()),
                                            baseline=self.baseline,
                                            preload=True)
            self.data_list[fn].event_id = self.new_data.event_id
        self.return_data = Epochs(self.new_data.raw_attr, self.data_list)
        self.destroy()

    def _get_result(self):
        return self.return_data

class SelectEvents(TopWindow):
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Select Events")
        self.preprocessed_data = preprocessed_data.copy()
        self.check_data()

        self.return_data = None

        tk.Label(self, text="Choose Events: ").pack()
        scrollbar  = tk.Scrollbar(self).pack(side="right", fill="y")
        self.listbox = tk.Listbox(self, selectmode="multiple", yscrollcommand=scrollbar)
        events_keys = list(self.preprocessed_data.event_id.keys())
        if len(events_keys) == 0:
            raise InitWindowValidateException(window=self, message="No Event")
        for each_item in range(len(events_keys)):
            self.listbox.insert(tk.END, events_keys[each_item])
        self.listbox.pack(padx=10, pady=10, expand=True, fill="both")
        tk.Button(self, text="Confirm", command=lambda: self._getEventID(), width=8).pack()

    def check_data(self):
        if type(self.preprocessed_data) != Raw:
            raise InitWindowValidateException(window=self, message="Invalid data")

    def _getEventID(self):
        new_event_id = {}
        self.listbox_idx = list(self.listbox.curselection())

        # Check if event is selected
        if len(self.listbox_idx) == 0:
            raise InitWindowValidateException(window=self, message="No valid data is loaded")

        for idx in self.listbox_idx:
            new_event_id[self.listbox.get(idx)] = self.preprocessed_data.event_id[self.listbox.get(idx)]

        new_events = {}
        for filename in self.preprocessed_data.raw_events.keys():
            new_events[filename] = mne.pick_events(self.preprocessed_data.raw_events[filename], include=list(new_event_id.values()))

        self.return_data = deepcopy(self.preprocessed_data)
        self.return_data.event_id = new_event_id
        self.return_data.raw_events = new_events
        self.destroy()

    def _get_result(self):
        return self.return_data