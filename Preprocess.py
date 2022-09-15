from Template import *

# ======================================================================= Channel
class Channel(TopWindow):
    def __init__(self, parent, title):
        super(Channel, self).__init__(parent, title)

        data_field = [ "ch_num"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Channel: ").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['ch_num'], bg="White").grid(row=6, column=1, sticky="w")

        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.data:
            if self.field_var['ch_num'].get() != "":
                select_ch = []
                for ch in self.field_var['ch_num'].get().replace(" ", "").split(','):
                    nums = ch.split(':')
                    if len(nums) > 1:
                        select_ch.extend(data.info['ch_names'][int(nums[0]) - 1:int(nums[1])])
                    else:
                        select_ch.extend([data.info['ch_names'][int(nums[0]) - 1]])
                data.pick_channels(select_ch)

        self.destroy()

    def _get_result(self):

        return self.parent.loaded_data


# ======================================================================= Filtering
class Filtering(TopWindow):
    def __init__(self, parent, title):
        super(Filtering, self).__init__(parent, title)

        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Lower pass-band edge: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['l_freq'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Upper pass-band edge: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['h_freq'], bg="White").grid(row=3, column=1, sticky="w")

        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.data:
            l_freq = float(self.field_var['l_freq'].get()) if self.field_var['l_freq'].get() != "" else None
            h_freq = float(self.field_var['h_freq'].get()) if self.field_var['h_freq'].get() != "" else None
            data.filter(l_freq=l_freq, h_freq=h_freq)

        self.destroy()

    def _get_result(self):

        return self.parent.loaded_data

# ======================================================================= Resample
class Resample(TopWindow):
    def __init__(self, parent, title):
        super(Resample, self).__init__(parent, title)
        data_field = [ "sfreq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Sampling Rate: ").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['sfreq'], bg="White").grid(row=6, column=1, sticky="w")

        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.data:
            if self.field_var['sfreq'].get() != "":
                data.resample(sfreq=float(self.field_var['sfreq'].get()))

        self.destroy()

    def _get_result(self):

        return self.parent.loaded_data


