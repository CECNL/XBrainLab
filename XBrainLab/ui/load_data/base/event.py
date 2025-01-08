import tkinter as tk
from tkinter import filedialog

import numpy as np

from XBrainLab.load_data import EventLoader

from ...base import TopWindow, ValidateException
from ...script import Script


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
        self.eventidframe = tk.LabelFrame(self, text="New Event ids:")

        tk.Button(self, text="Select file", command=self._load_event_file).grid(
            row=0, column=0, columnspan=2
        )
        tk.Label(self, text="New Event numbers: ").grid(
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
                self.load_script_history.add_cmd(
                    f"event_loader.read_txt({selected_file!r})"
                )
            elif '.mat' in selected_file:
                loaded_mat = self.event_loader.read_mat(selected_file)
                key_opt_val = [k for k in loaded_mat if not k.startswith('_')]
                if len(key_opt_val)<=1:
                    label_key = key_opt_val[0]
                else:
                    label_key = EventDictInfoSetter(self, loaded_mat, key_opt_val).get_result()
                label_list = self.event_loader.from_mat(loaded_mat[label_key])
                if not self.script_history.check_import("import scipy.io"):
                    self.load_script_history.add_import("import scipy.io")
                self.load_script_history.add_cmd("event_data = scipy.io.loadmat(filepath)")
                self.load_script_history.add_cmd(f"event_data = event_data[{label_key!r}]")
                self.load_script_history.add_cmd(f"event_loader.from_mat(event_data)")
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
        new_event_name = {e: self.new_event_name[e].get() for e in self.new_event_name}
        try:
            self.event_loader.create_event(new_event_name)
            self.script_history.add_script(self.load_script_history)
            self.script_history.add_cmd(
                f"event_loader.create_event(event_name_map={new_event_name!r})"
            )
        except Exception as e:
            raise ValidateException(self, str(e)) from e

        self.ret_script_history = self.script_history
        self.destroy()

    def _get_result(self):
        return self.event_loader.events, self.event_loader.event_id

    def _get_script_history(self):
        return self.ret_script_history
    
class EventDictInfoSetter(TopWindow):
    def __init__(self, parent, loaded_mat, key_opt_val):
        # ==== inits
        super().__init__(parent, "Select Field")
        self.loaded_mat = loaded_mat
        self.selected_key = None

        # generate options
        # key_opt_val = [k for k in loaded_mat if not k.startswith('_')]

        # generate vars
        self.event_key_trace = tk.StringVar(self)
        self.event_shape_view = tk.StringVar(self)
        self.event_key_trace.set('None')
        self.event_shape_view.set('None')
        self.event_key_trace.trace_add('write', self._shape_view_update)

        # ======== event key
        tk.Label(self, text="Event key: ").grid(row=0, column=0, sticky='w')
        tk.OptionMenu(
            self, self.event_key_trace, 'None', *key_opt_val
        ).grid(row=0, column=1, sticky='w')

        tk.Label(self, text="Value shape: ").grid(
            row=1, column=0, sticky='w'
        )
        tk.Label(self, textvariable=self.event_shape_view).grid(
            row=1, column=1
        )
        tk.Button(self, text="Confirm", command=self._key_confirm).grid(
            row=2, columnspan=2
        )
        self._shape_view_update()
    def _shape_view_update(self, *args):
        if self.event_key_trace.get() != 'None':
            self.event_shape_view.set(
                str(self.loaded_mat[self.event_key_trace.get()].shape)
            )
    def _key_confirm(self):
        self.selected_key = self.event_key_trace.get()
        self.destroy()
    def _get_result(self):
        # return self.loaded_mat[self.selected_key]
        return self.selected_key