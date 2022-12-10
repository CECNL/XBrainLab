from .base import PanelBase
import tkinter as tk
import tkinter.ttk as ttk

class TrainingStatusPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Status', **args)
        columns = ('Plan name', 'training trials', 'validation trials', 'testing trials', 'Progress')
        tree = ttk.Treeview(self, columns=columns, show='headings', selectmode=tk.BROWSE)
        for i in columns:
            tree.heading(i, text=i)
            tree.column(i, width=80, anchor=tk.CENTER)
        
        self.tree = tree

    def show_instruction(self):
        self.clear_panel()
        tk.Label(self, text='TODO: show steps').pack(expand=True)

    def show_panel(self):
        self.clear_panel()
        self.tree.pack(expand=True,fill=tk.BOTH)
        self.is_setup = True

    def update_panel(self, trainer):
        if not self.is_setup:
            self.show_panel()
        self.tree.delete(*self.tree.get_children())
        if not trainer:
            return
        plan_holders = trainer.get_training_plan_holders()
        for plan_holder in plan_holders:
            dataset = plan_holder.get_dataset()
            self.tree.insert("", 'end', values=[plan_holder.get_name(), dataset.get_train_len(), dataset.get_val_len(), dataset.get_test_len(), plan_holder.get_epoch_progress_text()])