import tkinter as tk
from tkinter import filedialog

from ...base import TopWindow, ValidateException

import numpy as np

class LoadEvent(TopWindow):
    def __init__(self, parent, raw):
        # ==== init
        super().__init__(parent, "Load events")
        self.raw = raw
        self.events = None
        self.event_id = None
        self.label_list = None
        self.minsize(260, 210)
        self.columnconfigure([0, 1], weight=1)
        self.rowconfigure([2], weight=1)

        self.event_num = tk.StringVar()
        self.new_event_name = {}
        self.eventidframe = tk.LabelFrame(self, text="Event ids:")

        tk.Button(self, text="Load file", command=self._load_event_file).grid(row=0, column=0, columnspan=2)
        tk.Label(self, text="Event numbers: ").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self, textvariable=self.event_num).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        self.eventidframe.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        tk.Button(self, text="Confirm", command=self._confirm).grid(row=3, column=0, columnspan=2, padx=5)
    
    def _load_event_file(self):
        selected_file = filedialog.askopenfilename(
            parent = self,
            filetypes = (
                ('text file', '*.txt'),
                #('mat file', '*mat'),
                #('text file', '*.lst'),
                #('text file', '*.eve'),
                #('binary file', '*.fif')
            )
        )
        if selected_file:
            label_list = []
            with open(selected_file, encoding='utf-8', mode='r') as fp:
                for line in fp.readlines():
                    label_list += [int(l.rstrip()) for l in line.split(' ')] # for both (n,1) and (1,n) of labels
                fp.close()
            self.label_list = label_list
            self.event_num.set(len(self.label_list))
            self.new_event_name = {k: tk.StringVar(self) for k in np.unique(label_list)}
            for child in self.eventidframe.winfo_children():
                child.destroy()
            for i, e in enumerate(np.unique(label_list)):
                self.new_event_name[e].set(e)
                tk.Label(self.eventidframe, text=e, width=10).grid(row=i, column=0)
        
    def _confirm(self, *args):
        if self.label_list:
            for e in self.new_event_name:
                if not self.new_event_name[e].get().strip():
                    raise ValidateException(self, "event name cannot be empty")
            event_id = {self.new_event_name[i].get(): i for i in list(set(self.label_list))}
            events = np.zeros((len(self.label_list), 3))
            events[:,0] = range(len(self.label_list))
            events[:,-1] = self.label_list

            if not self.raw.is_raw():
                if self.raw.get_epochs_length() != len(events):
                    raise ValidateException(self, f'Inconsistent number of events (got {len(events)})')

            self.events = events
            self.event_id = event_id
        else:
            raise ValidateException(self, "No label has been loaded.")
        self.destroy()

    def _get_result(self):
        return self.events, self.event_id