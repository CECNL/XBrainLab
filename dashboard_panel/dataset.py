from .base import PanelBase
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk

class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)

        self.preprocessed_data = None
        self.row_label = ["Data sessions: ", "Trials per session: ", "EEG channels: ", "Sampling rate: ", "Classes: "]
        self.row_var = {k:tk.StringVar() for k in self.row_label}
        
        
        for i in range(len(self.row_label)):
            tk.Label(self, text = self.row_label[i]).grid(row=i, column=0, sticky='w')
            tk.Label(self, textvariable= self.row_var[self.row_label[i]]).grid(row=i, column=1)
        self.update_panel(self.preprocessed_data)
            
    def update_panel(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        if preprocessed_data!= None and preprocessed_data.data!=[]:
            row_value = [len(self.preprocessed_data.data),\
                len(self.preprocessed_data.data[-1]),\
                0,\
                int(self.preprocessed_data.sfreq),\
                len(self.preprocessed_data.event_id)]
            if isinstance(preprocessed_data, Raw):
                row_value[1] = len(self.preprocessed_data.data) # raw has only 1 epoch
                row_value[2] = self.preprocessed_data.data[-1].get_data().shape[0]
            else:
                row_value[2] = self.preprocessed_data.data[-1].shape[0]

            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set(str(row_value[i]))
        else:
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set('None')



