from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
from ..base import InitWindowValidateException, ValidateException
import tkinter as tk
import tkinter.ttk as ttk

class Channel(TopWindow):
    command_label = "Channel"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Select Channel")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        self.return_data = None
        ch_names = self.preprocessed_data.data[0].ch_names

        tk.Label(self, text="Choose Channels: ").pack()
        scrollbar = tk.Scrollbar(self).pack(side="right", fill="y")
        self.listbox = tk.Listbox(self, selectmode="extended", yscrollcommand=scrollbar)
        for each_item in range(len(ch_names)):
            self.listbox.insert(tk.END, ch_names[each_item])
        self.listbox.pack(padx=10, pady=10, expand=True, fill="both")
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).pack()

    def check_data(self):
        if not(type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
            raise InitWindowValidateException(window=self, message="Invalid data")

    def _data_preprocess(self):
        self.select_channels = []
        for idx in list(self.listbox.curselection()):
            self.select_channels.append(self.listbox.get(idx))

        # Check if channel is selected
        if len(self.select_channels) == 0:
            raise InitWindowValidateException(window=self, message="No Channel Selected")

        self.data_list = []
        for data in self.preprocessed_data.data:
            self.data_list.append(data.copy().pick_channels(self.select_channels))

        self.return_data = self.preprocessed_data
        self.return_data.data = self.data_list
        self.destroy()

    def _get_result(self):
        return self.return_data

class Filtering(TopWindow):
    command_label = "Filtering"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Filtering")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        self.return_data = None
        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Lower pass-band edge: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['l_freq'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Upper pass-band edge: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['h_freq'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def check_data(self):
        if not(type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
            raise InitWindowValidateException(window=self, message="Invalid data")

    def _data_preprocess(self):
        # Check if input is empty
        if self.field_var['l_freq'].get() == "" and self.field_var['h_freq'].get() == "":
            raise InitWindowValidateException(window=self, message="No Input")

        self.data_list = []
        for data in self.preprocessed_data.data:
            l_freq = float(self.field_var['l_freq'].get()) if self.field_var['l_freq'].get() != "" else None
            h_freq = float(self.field_var['h_freq'].get()) if self.field_var['h_freq'].get() != "" else None
            self.data_list.append(data.copy().filter(l_freq=l_freq, h_freq=h_freq))

        self.return_data = self.preprocessed_data
        self.return_data.data = self.data_list
        self.destroy()

    def _get_result(self):
        return self.return_data

class Resample(TopWindow):
    command_label = "Resample"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Resample")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        self.return_data = None
        data_field = [ "sfreq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Sampling Rate: ").grid(row=6, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['sfreq'], bg="White").grid(row=6, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def check_data(self):
        if not(type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
            raise InitWindowValidateException(window=self, message="Invalid data")

    def _data_preprocess(self):
        # Check Input is Valid
        if self.field_var['sfreq'].get() == "":
            raise InitWindowValidateException(window=self, message="No Input")
        elif float(self.field_var['sfreq'].get()) < 0.0:
            raise InitWindowValidateException(window=self, message="Input value invalid")

        self.data_list = []
        for data in self.preprocessed_data.data:
            self.data_list.append(data.copy().resample(sfreq=float(self.field_var['sfreq'].get())))

        self.return_data = self.preprocessed_data
        self.return_data.sfreq = self.field_var['sfreq'].get()
        self.return_data.data = self.data_list
        self.destroy()

    def _get_result(self):
        return self.return_data


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
            raise InitWindowValidateException(self, 'No valid data were loaded.')
        if self.preprocessed_data.event_ids == {}:
            raise InitWindowValidateException(self, 'Lacking events in loaded data.')
    
    def _confirm(self, confirm_bool=0):
        if confirm_bool ==1:
            # get from entry to new_event dict
            for i in range(len(self.old_event)):
                if len(set([v.get() for v in self.new_event_name.values()])) < len(self.old_event):
                    raise ValidateException(window=self, message="Duplicate event name.")

            # update parent event data
            for k in self.old_event.keys():
                self.new_event[self.new_event_name[k].get()] = self.old_event[k]
            self.preprocessed_data.event_id = self.new_event
            
            if isinstance(self.preprocessed_data, Raw): # Raw
                self.preprocessed_data.event_id = self.new_event
            else:
                for mne_struct in self.preprocessed_data.epoch_data:
                    mne_struct.event_id = self.new_event
        
        #self.preprocessed_data.inspect()
        self.destroy()
    def _get_result(self):
        return self.preprocessed_data