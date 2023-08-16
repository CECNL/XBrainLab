import tkinter as tk
import numpy as np

from .base import PreprocessBase
from ..base import ValidateException, InitWindowValidateException

from XBrainLab import preprocessor as Preprocessor

class EditEventNames(PreprocessBase):
    #  menu state disable
    command_label = "Edit Event Name"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.EditEventName(preprocessed_data_list)
        super().__init__(parent, "Edit Event Name", preprocessor)
        self.old_event = set()
        for preprocessed_data in preprocessor.get_preprocessed_data_list():
            _, event_id = preprocessed_data.get_event_list()
            self.old_event.update(event_id)

        self.new_event_name = {k: tk.StringVar() for k in self.old_event}
        
        self.rowconfigure([0], weight=1)
        self.columnconfigure([0, 1], weight=1)

        eventidframe = tk.LabelFrame(self, text="Event Names:")
        for i, e in enumerate(self.old_event):
            self.new_event_name[e].set(e)
            tk.Label(eventidframe, text=e, width=10).grid(row=i, column=0)
            tk.Entry(eventidframe, textvariable=self.new_event_name[e], width=10).grid(row=i, column=1)
        if i:
            eventidframe.rowconfigure(list(range(i+1)), weight=1)
        eventidframe.columnconfigure([0, 1], weight=1)

        eventidframe.grid(row=0, column=0, columnspan=2, sticky='news', padx=5, pady=5)
        tk.Button(self, text="Cancel", command=self.destroy).grid(row=1, column=0)
        tk.Button(self, text="Confirm", command=self._confirm).grid(row=1, column=1)

    def _confirm(self):
        # update parent event data
        for _, new_event_name in self.new_event_name.items():
            if not new_event_name.get().strip():
                raise ValidateException(self, "Event name cannot be empty")
        
        try:
            new_event_name = {e: self.new_event_name[e].get() for e in self.new_event_name}
            self.return_data = self.preprocessor.data_preprocess(new_event_name)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        self.script_history.add_cmd(f'new_event_name={repr(new_event_name)}')
        self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.EditEventName, new_event_name=new_event_name)')
        self.ret_script_history = self.script_history
        
        self.destroy()

class EditEventIds(PreprocessBase):
    #  menu state disable
    command_label = "Edit Event Ids"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.EditEventId(preprocessed_data_list)
        super().__init__(parent, "Edit Event Ids", preprocessor)
        self.old_event, self.old_event_id = np.zeros((0, 3)), dict()
        for preprocessed_data in preprocessor.get_preprocessed_data_list():
            event, event_id = preprocessed_data.get_event_list()
            self.old_event = np.concatenate((self.old_event, event), axis=0)
            self.old_event_id.update(event_id)

        self.new_event_id = {v: tk.StringVar() for v in self.old_event_id.values()}
        
        self.rowconfigure([0], weight=1)
        self.columnconfigure([0, 1], weight=1)

        eventidframe = tk.LabelFrame(self, text="Event Ids:")
        for i, e in enumerate(self.old_event_id.values()):
            self.new_event_id[e].set(e)
            tk.Label(eventidframe, text=e, width=10).grid(row=i, column=0)
            tk.Entry(eventidframe, textvariable=self.new_event_id[e], width=10).grid(row=i, column=1)
        if i:
            eventidframe.rowconfigure(list(range(i+1)), weight=1)
        eventidframe.columnconfigure([0, 1], weight=1)

        eventidframe.grid(row=0, column=0, columnspan=2, sticky='news', padx=5, pady=5)
        tk.Button(self, text="Cancel", command=self.destroy).grid(row=1, column=0)
        tk.Button(self, text="Confirm", command=self._confirm).grid(row=1, column=1)

    def _confirm(self):
        # update parent event data
        for _, new_event_id in self.new_event_id.items():
            if not new_event_id.get().strip():
                raise ValidateException(self, "Event id cannot be empty")
        
        try:
            new_event_id = {e: int(self.new_event_id[e].get()) for e in self.new_event_id.keys()}
            self.return_data = self.preprocessor.data_preprocess(new_event_id)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        self.script_history.add_cmd(f'new_event_id={repr(new_event_id)}')
        self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.EditEventId, new_event_ids=new_event_id)')
        self.ret_script_history = self.script_history
        
        self.destroy()