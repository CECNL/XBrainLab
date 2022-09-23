from .base import PanelBase
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk

class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)

        self.preprocessed_data = None
        self.row_label = ["Data sessions: ", "Avg trials per session: ", "EEG channels: ", "Sampling rate: ", "Classes: "]
        self.row_var = {k:tk.StringVar() for k in self.row_label}
        [self.row_var[k].set('None') for k in self.row_var.keys()]
        
        
        for i in range(len(self.row_label)):
            tk.Label(self, text = self.row_label[i]).grid(row=i, column=0, sticky='w')
            tk.Label(self, textvariable= self.row_var[self.row_label[i]]).grid(row=i, column=1)
        self.update_panel(self.preprocessed_data)
            
    def update_panel(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        if preprocessed_data!= None:   
            row_value = [len(self.preprocessed_data.mne_data),\
                0,\
                0,\
                int(self.preprocessed_data.sfreq),\
                len(self.preprocessed_data.event_id)]
            if isinstance(preprocessed_data, Raw):
                event_sum = 0
                for k in self.preprocessed_data.raw_events.keys():
                    event_sum += self.preprocessed_data.raw_events[k].shape[0]
                    row_value[2] = len(self.preprocessed_data.mne_data[k].ch_names)
                row_value[1] = int(event_sum/len(self.preprocessed_data.mne_data))
            else:
                epoch_sum = 0
                for k in self.preprocessed_data.mne_data.keys():
                    epoch_sum += len(self.preprocessed_data.mne_data[k])
                    row_value[2] = len(self.preprocessed_data.mne_data[k].ch_names)
                row_value[1] = int(epoch_sum/len(self.preprocessed_data.mne_data))
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set(str(row_value[i])) 
        else:
            for i in range(len(self.row_label)):
                self.row_var[self.row_label[i]].set('None')



