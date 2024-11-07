import tkinter as tk

from .base import PanelBase


StandardSaliencyParam = {
    'nt_samples': 5,
    'nt_samples_batch_size': None,
    'stdevs': 1.0
}


class ExplanationSettingPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Explanation Setting', **args)
        frame = tk.Frame(self)
        tk.Label(frame, text='Computation parameters').grid(row=0, column=0, sticky='e', padx=10)


        self.epoch_label = tk.Label(frame, text='Default')

        self.epoch_label.grid(row=0, column=1)

        frame.pack(expand=True)
        self.frame = frame

    def show_panel(self):
        self.clear_panel()
        self.frame.pack(expand=True)

        self.is_setup = True

    def update_panel(self, update_saliency_param):
        if not self.is_setup:
            self.show_panel()
        self.epoch_label.config(text='Not set')

        update_text = 'Default'

        if update_saliency_param is not None:
            for algo, params in update_saliency_param.items():
                if update_text != 'Default':
                    break
                for param, value in params.items():
                    if value != StandardSaliencyParam[param]:
                        update_text = 'Custom'
                        break
                
            self.epoch_label.config(text=update_text)

