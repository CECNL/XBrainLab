from .base import PreprocessBase
from ..base import ValidateException, InitWindowValidateException
import tkinter as tk

class EditEvent(PreprocessBase):
    #  menu state disable
    command_label = "Edit Event"
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Edit Event", preprocessed_data_list)
        self.old_event = set()
        for preprocessed_data in self.preprocessed_data_list:
            _, event_id = preprocessed_data.get_event_list()
            self.old_event.update(event_id)

        self.new_event_name = {k: tk.StringVar() for k in self.old_event}
        
        self.rowconfigure([0], weight=1)
        self.columnconfigure([0, 1], weight=1)

        eventidframe = tk.LabelFrame(self, text="Event ids:")
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

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise InitWindowValidateException(window=self, message=f"Event name can only be edited for epoched data")

    def get_preprocess_desc(self, event_id):
        return f"Update {len(event_id)} event"

    def _confirm(self):
        # update parent event data
        for _, new_event_name in self.new_event_name.items():
            if not new_event_name.get().strip():
                raise ValidateException(self, "Event name cannot be empty")
        
        for preprocessed_data in self.preprocessed_data_list:
            events, event_id = preprocessed_data.get_event_list()
            new_event_id = {}
            for e in event_id:
                new_event_id[self.new_event_name[e].get()] = event_id[e]
            preprocessed_data.set_event(events, new_event_id)
            preprocessed_data.add_preprocess(self.get_preprocess_desc(new_event_id))
        
        self.return_data = self.preprocessed_data_list
        self.destroy()