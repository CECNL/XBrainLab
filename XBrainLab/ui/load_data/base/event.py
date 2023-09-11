import numpy as np
import tkinter as tk
from tkinter import filedialog

from ...base import TopWindow, ValidateException
from ...script import Script

from XBrainLab.load_data import EventLoader

class LoadEvent(TopWindow):
    def __init__(self, parent, raw):
        # ==== init
        super().__init__(parent, "Load events")
        self.event_loader = EventLoader(raw)
        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.load_data import EventLoader")
        self.script_history.add_cmd("event_loader = EventLoader(raw_data)")
        self.load_script_history = Script()
        self.ret_script_history = None

        self.minsize(260, 210)
        self.columnconfigure([0, 1], weight=1)
        self.rowconfigure([2], weight=1)

        self.event_num = tk.StringVar()
        self.new_event_name = {}
        self.eventidframe = tk.LabelFrame(self, text="Event ids:")

        tk.Button(self, text="Load file", command=self._load_event_file).grid(
            row=0, column=0, columnspan=2
        )
        tk.Label(self, text="Event numbers: ").grid(
            row=1, column=0, sticky='w', padx=5, pady=5
        )
        tk.Label(self, textvariable=self.event_num).grid(
            row=1, column=1, sticky='w', padx=5, pady=5
        )
        self.eventidframe.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        tk.Button(self, text="Confirm", command=self._confirm).grid(
            row=3, column=0, columnspan=2, padx=5
        )
    
    def _load_event_file(self):
        selected_file = filedialog.askopenfilename(
            parent = self,
            filetypes = (
                ('mat file', '*mat'),
                ('text file', '*.txt'),
                #('text file', '*.lst'),
                #('text file', '*.eve'),
                #('binary file', '*.fif')
            )
        )
        if selected_file: # event array length incompatiable not handled
            if '.txt' in selected_file:
                label_list = self.event_loader.read_txt(selected_file)
                self.load_script_history.set_cmd(
                    f"event_loader.read_txt({repr(selected_file)})"
                )
            elif '.mat' in selected_file:
                label_list = self.event_loader.read_mat(selected_file)
                self.load_script_history.set_cmd(
                    f"event_loader.read_mat({repr(selected_file)})"
                )
            else:
                tk.messagebox.showwarning(
                    parent=self,
                    title="Warning",
                    message="Event file format not supported."
                )
            
            self.event_num.set(len(label_list))
            self.new_event_name = {
                k: tk.StringVar(self) for k in np.unique(label_list)
            }
            for child in self.eventidframe.winfo_children():
                child.destroy()
            for i, e in enumerate(np.unique(label_list)):
                self.new_event_name[e].set(e)
                tk.Label(self.eventidframe, text=e, width=10).grid(
                    row=i, column=0
                )
        
    def _confirm(self, *args):
        new_event_name = {e:self.new_event_name[e].get() for e in self.new_event_name}
        try:
            self.event_loader.create_event(new_event_name)
            self.script_history.add_script(self.load_script_history)
            self.script_history.add_cmd(
                f"event_loader.create_event(event_name_map={repr(new_event_name)})"
            )
        except Exception as e:
            raise ValidateException(self, str(e))
        
        self.ret_script_history = self.script_history
        self.destroy()

    def _get_result(self):
        return self.event_loader.events, self.event_loader.event_id

    def _get_script_history(self):
        return self.ret_script_history