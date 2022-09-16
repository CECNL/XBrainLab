from ..base.top_window import TopWindow
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
import mne

class TimeEpoch(TopWindow):
    command_label = "Time Epoch"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Time Epoch")
        self.preprocessed_data = preprocessed_data
        data_field = ["select_events", "tmin", "tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Choose Events: ").grid(row=1, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['select_events'], bg="White").grid(row=1, column=1, sticky="w")
        tk.Button(self, text="...", command=lambda: self._choose_events(), width=8).grid(row=1, column=2)
        tk.Label(self, text="Epoch limit start: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmin'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Epoch limit end: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmax'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda: self._extract_epoch(), width=8).grid(row=7, columnspan=3)

    def _choose_events(self):
        events_keys = list(self.parent.loaded_data.ret_val.event_id.keys())
        self.selectedEvents, self.event_name = SelectEvents(self, events_keys).get_result()
        self.field_var['select_events'].set(",".join(self.event_name))

    def _extract_epoch(self):
        baseline = BaselineRemoval(self, "Baseline Removal", self.field_var['tmin'].get()).get_result()

        self.data_list = []
        for data in self.parent.loaded_data.ret_val.data:
            self.data_list.append(mne.Epochs(data, self.selectedEvents, tmin=float(self.field_var['tmin'].get()), tmax=float(self.field_var['tmax'].get()), baseline=baseline))
        self.parent.loaded_data.ret_val.data = self.data_list
        self.destroy()

    def _get_result(self):
        return self.preprocessed_data

class WindowEpoch(TopWindow):
    command_label = "Window Epoch"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Window Epoch")
        self.preprocessed_data = preprocessed_data
        data_field = ["duration", "overlap"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Duration of each epoch: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['duration'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Overlap between epochs: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['overlap'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda: self._extract_epoch(), width=8).grid(row=7, columnspan=2)

    def _extract_epoch(self):
        baseline = BaselineRemoval(self, "Baseline Removal").get_result()

        self.data_list = []
        for data in self.parent.loaded_data.ret_val.data:
            overlap = 0.0 if self.field_var['overlap'].get() == "" else float(self.field_var['overlap'].get())
            epoch = mne.make_fixed_length_epochs(data, duration=float(self.field_var['duration'].get()), overlap=overlap)
            if baseline is not None:
                epoch.average().apply_baseline(baseline)
            self.data_list.append(epoch)

        self.parent.loaded_data.ret_val.data = self.data_list
        self.destroy()

    def _get_result(self):
        return self.preprocessed_data

class BaselineRemoval(TopWindow):
    def __init__(self, parent, preprocessed_data, tmin = None):
        super().__init__(parent, "Baseline Removal")
        self.baseline = None
        data_field = ["tmin", "tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Baseline latency range").grid(row=1, columnspan=2, sticky="w")
        tk.Label(self, text="min: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmin'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="max: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmax'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Label(self, text="Note: press Cancel if you don't want to do baseline removal.").grid(row=5, columnspan=2, sticky="w")
        tk.Button(self, text="Cancel", command=lambda: self.cancel(), width=8).grid(row=7, column=0)
        tk.Button(self, text="Confirm", command=lambda: self.confirm(), width=8).grid(row=7, column=1)

        if tmin is not None:
            self.field_var['tmin'].set(tmin)
            self.field_var['tmax'].set("0")

    def confirm(self):
        baseline_tmin = float(self.field_var['tmin'].get()) if self.field_var['tmin'].get() != "" else None
        baseline_tmax = float(self.field_var['tmax'].get()) if self.field_var['tmax'].get() != "" else None
        self.baseline = (baseline_tmin, baseline_tmax)
        self.destroy()

    def cancel(self):
        self.destroy()

    def _get_result(self):
        return self.baseline

class SelectEvents(TopWindow):
    def __init__(self, parent, events_keys):
        super().__init__(parent, "Select Events")

        tk.Label(self, text="Choose Events: ").pack()
        scrollbar  = tk.Scrollbar(self).pack(side = "right", fill = "y")
        self.listbox = tk.Listbox(self, selectmode="multiple", yscrollcommand=scrollbar)
        for each_item in range(len(events_keys)):
            self.listbox.insert(tk.END, events_keys[each_item])
        self.listbox.pack(padx=10, pady=10, expand=True, fill="both")
        tk.Button(self, text="Confirm", command=lambda: self._getEventID(), width=8).pack()

    def _getEventID(self):
        new_event_id = {}
        select_events = []
        self.events_name = []

        listbox_idx = list(self.listbox.curselection())
        for idx in listbox_idx:
            event_id = self.parent.parent.loaded_data.ret_val.event_id[self.listbox.get(idx)]
            self.events_name.append(self.listbox.get(idx))
            select_events.append(event_id)
            new_event_id[self.listbox.get(idx)] = event_id

        try:
            old_events = mne.find_events(self.parent.parent.loaded_data.ret_val.data[0])
        except:
            try:
                old_events = mne.events_from_annotations(self.parent.parent.loaded_data.ret_val.data[0])
            except (ValueError, TypeError) as err:
                print(err)
        self.new_events = mne.pick_events(old_events[0], include=select_events)

        new_label = []
        for label in self.parent.parent.loaded_data.ret_val.label:
            new_label.append([value for value in label if value in select_events])
        self.parent.parent.loaded_data.ret_val.label = new_label
        self.parent.parent.loaded_data.ret_val.event_id = new_event_id

        self.destroy()

    def _get_result(self):
        return self.new_events, self.events_name