from Template import *

# ======================================================================= Time Epoch
class TimeEpoch(TopWindow):
    def __init__(self, parent, title):
        super(TimeEpoch, self).__init__(parent, title)

        data_field = ["tmin", "tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Epoch limit start: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmin'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Epoch limit end: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmax'], bg="White").grid(row=3, column=1, sticky="w")

        tk.Button(self, text="Confirm", command=lambda: self._extract_epoch(), width=8).grid(row=7, columnspan=2)

    def _extract_epoch(self):
        baseline = BaselineRemoval(self, "Baseline Removal", self.field_var['tmin'].get()).get_result()

        self.data_list = []
        for data in self.parent.loaded_data.data:
            events = []
            try:
                events = mne.find_events(data)
            except:
                try:
                    events = mne.events_from_annotations(data)
                except (ValueError, TypeError) as err:
                    print(err)
            self.data_list.append(mne.Epochs(data, events[0], tmin=float(self.field_var['tmin'].get()), tmax=float(self.field_var['tmax'].get()), baseline=baseline))

        self.parent.loaded_data.data = self.data_list

        self.destroy()

    def _get_result(self):

        return self.parent.loaded_data


# ======================================================================= Window Epoch
class WindowEpoch(TopWindow):
    def __init__(self, parent, title):
        super(WindowEpoch, self).__init__(parent, title)

        data_field = ["duration", "overlap"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Duration of each epoch: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['duration'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Overlap between epochs: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['overlap'], bg="White").grid(row=3, column=1, sticky="w")

        tk.Button(self, text="Confirm", command=lambda: self._extract_epoch(), width=8).grid(row=7, columnspan=2)

    def _extract_epoch(self):
        self.data_list = []
        for data in self.parent.loaded_data.data:
            overlap = 0.0 if self.field_var['overlap'].get() == "" else float(self.field_var['overlap'].get())
            self.data_list.append(mne.make_fixed_length_epochs(data, duration=float(self.field_var['duration'].get()), overlap=overlap))

        self.parent.loaded_data.data = self.data_list
        self.destroy()

    def _get_result(self):

        return self.parent.loaded_data


# ======================================================================= Baseline Removal
class BaselineRemoval(TopWindow):
    def __init__(self, parent, title, tmin = None):
        super(BaselineRemoval, self).__init__(parent, title)

        self.doRemoval = True if tmin is None else False

        tk.Label(self, text="Baseline latency range").grid(row=1, columnspan=2, sticky="w")

        data_field = ["tmin", "tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="min: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmin'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="max: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['tmax'], bg="White").grid(row=3, column=1, sticky="w")

        if not self.doRemoval:
            self.field_var['tmin'].set(tmin)
            self.field_var['tmax'].set("0")

        tk.Label(self, text="Note: press Cancel if you don't want to do baseline removal.").grid(row=5, columnspan=2, sticky="w")

        tk.Button(self, text="Cancel", command=lambda: self.cancel(), width=8).grid(row=7, column=0)
        tk.Button(self, text="Confirm", command=lambda: self.confirm(), width=8).grid(row=7, column=1)

    def confirm(self):
        if self.doRemoval:
           for data in self.parent.loaded_data.data:
                tmin_input = float(self.field_var['tmin'].get()) if self.field_var['tmin'].get() != "" else None
                tmax_input = float(self.field_var['tmax'].get()) if self.field_var['tmax'].get() != "" else None
                data.average().apply_baseline((tmin_input, tmax_input))
        else:
            self.baseline = (float(self.field_var['tmin'].get()), float(self.field_var['tmax'].get()))
        self.destroy()

    def cancel(self):
        self.baseline = None
        self.destroy()

    def _get_result(self):
        if self.doRemoval:
            return self.parent.loaded_data
        else:
            return self.baseline