from .base import PanelBase
from ..dataset.data_holder import Raw, Epochs
from ..base import TopWindow
import tkinter as tk

class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)

        self.preprocessed_data = None
        self.row_label = ["Data sessions: ","Session Details: ","Classes: "]
        self.row_var = {k:tk.StringVar() for k in self.row_label}
        [self.row_var[k].set('None') for k in self.row_var.keys()]
        self.btn = tk.Button(self, text="View", command=lambda:self.attr_detail(), state=tk.DISABLED)
        
        
        for i in range(len(self.row_label)):
            tk.Label(self, text = self.row_label[i]).grid(row=i, column=0, sticky='w')
            if i != 1:
                tk.Label(self, textvariable= self.row_var[self.row_label[i]]).grid(row=i, column=1)
            else:
                self.btn.grid(row=i, column=1)
        self.update_panel(self.preprocessed_data)
    
    def attr_detail(self):
        win = _attr_table(self, self.preprocessed_data)

    def update_panel(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        if preprocessed_data!= None:  
            self.btn['state'] = tk.NORMAL
            row_value = [len(self.preprocessed_data.mne_data),\
                0,\
                len(self.preprocessed_data.event_id)]
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set(str(row_value[i])) 
        else:
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set('None')

class _attr_table(TopWindow):
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Subjects and sessions")
        self.subjects = {}
        self.sessions = {}
        if preprocessed_data == None:
            return False

        if isinstance(preprocessed_data, Raw):
            self.subjects = preprocessed_data.subjects.copy()
            self.sessions = preprocessed_data.sessions.copy()
        else:
            self.subjects = preprocessed_data.subject_map.copy()
            self.sessions = {k:0 for k in self.subjects.values()}
            for fn in self.preprocess_data.epoch_attr.keys():
                self.sessions[self.preprocess_data.epoch_attr[fn][0]] += 1
        
        tk.Label(self, text="Subject").grid(row=0, column=0, sticky='w')
        tk.Label(self, text="No. Sessions").grid(row=0, column=1, sticky='w')
        for i in range(len(self.subjects)):
            tk.Label(self, text= self.subjects[i]).grid(row=i+1, column=0)
            tk.Label(self, text= str(self.sessions[self.subjects[i]])).grid(row=i+1, column=1)
    def _get_result(self):
        return True

