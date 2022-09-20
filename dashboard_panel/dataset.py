from .base import PanelBase
import tkinter as tk

class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)

        self.preprocessed_data = None
        self.row_label = ["Data sessions: ", "Epochs per session: ", "EEG channels: ", "Sampling rate: ", "Classes: "]
        self.row_var = {k:tk.StringVar() for k in self.row_label}
        
        
        for i in range(len(self.row_label)):
            tk.Label(self, text = self.row_label[i]).grid(row=i, column=0, sticky='w')
            tk.Label(self, textvariable= self.row_var[self.row_label[i]]).grid(row=i, column=1)
        self.update_panel(self.preprocessed_data)
            
    def update_panel(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        if preprocessed_data!= None:
            row_value = [len(self.preprocessed_data.data),\
                len(self.preprocessed_data.data[-1]),\
                self.preprocessed_data.data[-1].info['nchan'],\
                int(self.preprocessed_data.data[-1].info['sfreq']),\
                len(self.preprocessed_data.label_map)]
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set(str(row_value[i]))
        else:
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set('None')



