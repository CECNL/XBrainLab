from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
import tkinter.ttk as ttk

class Channel(TopWindow):
    command_label = "Channel"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Select Channel")
        self.preprocessed_data = preprocessed_data
        data_field = [ "ch_num"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Channel: ").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['ch_num'], bg="White").grid(row=6, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.ret_val.data:
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
        return self.preprocessed_data

class Filtering(TopWindow):
    command_label = "Filtering"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Filtering")
        self.preprocessed_data = preprocessed_data
        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Lower pass-band edge: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['l_freq'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Upper pass-band edge: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['h_freq'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.ret_val.data:
            l_freq = float(self.field_var['l_freq'].get()) if self.field_var['l_freq'].get() != "" else None
            h_freq = float(self.field_var['h_freq'].get()) if self.field_var['h_freq'].get() != "" else None
            data.filter(l_freq=l_freq, h_freq=h_freq)
        self.destroy()

    def _get_result(self):
        return self.preprocessed_data

class Resample(TopWindow):
    command_label = "Resample"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Resample")
        self.preprocessed_data = preprocessed_data
        data_field = [ "sfreq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Sampling Rate: ").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['sfreq'], bg="White").grid(row=6, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def _data_preprocess(self):
        for data in self.parent.loaded_data.ret_val.data:
            if self.field_var['sfreq'].get() != "":
                data.resample(sfreq=float(self.field_var['sfreq'].get()))
        self.destroy()

    def _get_result(self):
        return self.preprocessed_data


class EditEvent(TopWindow):
    #  menu state disable
    command_label = "Edit Event"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Edit Event")
        self.old_event = preprocessed_data.event_id
        self.preprocessed_data = preprocessed_data
        self.new_event_name = {k:tk.StringVar() for k in self.old_event.keys()}
        self.new_event = {}
        self.check_data()

        eventidframe = tk.LabelFrame(self, text="Event ids:").grid(row=0, column=0, columnspan=2, sticky='w')
        i = 0
        for k,v in self.old_event.items():
            self.new_event_name[k].set(k)
            tk.Entry(eventidframe, textvariable=self.new_event_name[k]).grid(row=i, column=0)
            tk.Label(eventidframe, text=str(v)).grid(row=i, column=1)
            i += 1
        
        tk.Button(self, text="Cancel", command=lambda:self._confirm(0)).grid(row=1, column=0)
        tk.Button(self, text="Confirm", command=lambda:self._confirm(1)).grid(row=1, column=1)
    
    def check_data(self):
        if not any([ isinstance(self.preprocessed_data, Raw), isinstance(self.preprocessed_data, Epochs)]):
            raise InitWindowValidateException(self, 'No validate data is loaded.')
        if self.preprocessed_data.event_ids == {}:
            raise InitWindowValidateException(self, 'Lacking events in loaded data.')
    
    def _confirm(self, confirm_bool=0):
        if confirm_bool ==1:
            # get from entry to new_event dict
            for i in range(len(self.old_event)):
                if len(set([v.get() for v in self.new_event_name.values()])) < len(self.old_event):
                    raise ValidateException("Duplicate event name.")

            # update parent event data
            for k in self.old_event.keys():
                self.new_event[self.new_event_name[k].get()] = self.old_event[k]
            self.preprocessed_data.event_id = self.new_event
            
            if isinstance(self.preprocessed_data, Raw): # Raw
                self.preprocessed_data.event_id = self.new_event
            else:
                for mne in self.preprocessed_data.epoch_data:
                    mne.event_id = self.new_event
        
        #self.preprocessed_data.inspect()
        self.destroy()
    def _get_result(self):
        return self.preprocessed_data