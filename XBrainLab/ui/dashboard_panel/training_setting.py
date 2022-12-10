from .base import PanelBase
import tkinter as tk

class TrainingSettingPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Setting', **args)
        frame = tk.Frame(self)
        tk.Label(frame, text='Epoch').grid(row=0, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Batch size').grid(row=1, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Learning rate').grid(row=2, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Optimizer').grid(row=3, column=0, sticky='e', padx=10)
        tk.Label(frame, text='device').grid(row=4, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Output Directory').grid(row=5, column=0, sticky='e', padx=10)
        tk.Label(frame, text='CheckPoint epoch').grid(row=6, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Evaluation').grid(row=7, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Repeat Number').grid(row=8, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Model').grid(row=9, column=0, sticky='e', padx=10)

        self.epoch_label = tk.Label(frame, text='Epoch')
        self.bs_label = tk.Label(frame, text='Batch size')
        self.lr_label = tk.Label(frame, text='Learning rate')
        self.optim_label = tk.Label(frame, text='Optimizer')
        self.device_label = tk.Label(frame, text='device')
        self.output_label = tk.Label(frame, text='Output Directory')
        self.checkpoint_label = tk.Label(frame, text='CheckPoint epoch')
        self.eval_label = tk.Label(frame, text='Evaluation')
        self.repeat_label = tk.Label(frame, text='Repeat Number')
        self.model_label = tk.Label(frame, text='Model')

        self.epoch_label.grid(row=0, column=1)
        self.bs_label.grid(row=1, column=1)
        self.lr_label.grid(row=2, column=1)
        self.optim_label.grid(row=3, column=1)
        self.device_label.grid(row=4, column=1)
        self.output_label.grid(row=5, column=1)
        self.checkpoint_label.grid(row=6, column=1)
        self.eval_label.grid(row=7, column=1)
        self.repeat_label.grid(row=8, column=1)
        self.model_label.grid(row=9, column=1)
        
        frame.pack(expand=True)
        self.frame = frame

    def show_instruction(self):
        self.clear_panel()
        tk.Label(self, text='TODO: show steps').pack(expand=True)

    def show_panel(self):
        self.clear_panel()
        self.frame.pack(expand=True)

        self.is_setup = True

    def update_panel(self, model_holder, training_option):
        if not self.is_setup:
            self.show_panel()

        self.epoch_label.config(text='Not set')
        self.bs_label.config(text='Not set')
        self.lr_label.config(text='Not set')
        self.optim_label.config(text='Not set')
        self.device_label.config(text='Not set')
        self.output_label.config(text='Not set')
        self.checkpoint_label.config(text='Not set')
        self.eval_label.config(text='Not set')
        self.repeat_label.config(text='Not set')
        self.model_label.config(text='Not set')
        
        if model_holder:
            self.model_label.config(text=model_holder.get_model_desc_str())
        
        if training_option:
            self.epoch_label.config(text=training_option.epoch)
            self.bs_label.config(text=training_option.bs)
            self.lr_label.config(text=training_option.lr)
            self.optim_label.config(text=training_option.get_optim_desc_str())
            self.device_label.config(text=training_option.get_device_name())
            self.output_label.config(text=training_option.output_dir)
            self.checkpoint_label.config(text=training_option.checkpoint_epoch)
            self.eval_label.config(text=training_option.evaluation_option.value)
            self.repeat_label.config(text=training_option.repeat_num)
            