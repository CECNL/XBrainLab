import tkinter as tk
import tkinter.ttk as ttk
from ..base import TopWindow
from enum import Enum
import numpy as np 

class Metric(Enum):
    ACC = 'Accuracy (%)'
    KAPPA = 'kappa value'

class EvaluationTableWindow(TopWindow):
    command_label = 'Performance Table'
    def __init__(self, parent, trainers):
        super().__init__(parent, self.command_label)
        self.trainers = trainers
        if not self.check_data():
            return
        # init data
        metric_list = [i.value for i in Metric]

        # gui
        ## option menu
        selected_metric = tk.StringVar(self)
        selected_metric.set(metric_list[0])
        selected_metric.trace('w', lambda *args: self.update_loop(loop=False))
        metric_opt = tk.OptionMenu(self, selected_metric, *metric_list)
        ## tree
        
        columns = [plan.get_name() for plan in self.trainers[0].get_plans()] + ['Average']
        tree = ttk.Treeview(self, columns=columns, selectmode=tk.BROWSE)
        tree.heading('#0', text='Plan Name')
        for i in columns:
            tree.heading(i, text=i)
        for trainer in self.trainers:
            tree.insert("", 'end', iid=trainer.get_name(), values=(), text=trainer.get_name())
        tree.insert("", 'end', values=(), text='Average')

        metric_opt.grid(row=0, column=0)
        tree.grid(row=1, column=0, sticky='news')

        self.selected_metric = selected_metric
        self.tree = tree
        self.update_loop()

    def check_data(self):
        if type(self.trainers) != list:
            self.valid = False
            self.withdraw()
            tk.messagebox.showerror(parent=self.master, title='Error', message='No valid training plan is generated')
            self.destroy()
            return False
        return True

    def update_loop(self, loop=True):
        if self.winfo_exists() == 0:
            return
        total_values = []
        for trainer in self.trainers:
            values = ()
            if self.selected_metric.get() == Metric.ACC.value:
                values = [plan.get_acc() for plan in trainer.get_plans()]
            elif self.selected_metric.get() == Metric.KAPPA.value:
                values = [plan.get_kappa() for plan in trainer.get_plans()]
            total_values.append(values)
            values_list = [i for i in values if i is not None]
            if len(values_list) > 0:
                values = values + [sum(values_list) / len(values_list)]
            self.tree.item(trainer.get_name(), values=values)
        # average
        avg_values = []
        if len(total_values) > 0:
            for i in range(len(total_values[0])):
                values_list = []
                for values in total_values:
                    if values[i] is not None:
                        values_list.append(values[i])
                if len(values_list) > 0:
                    avg_values.append(sum(values_list) / len(values_list))
        self.tree.item(self.tree.get_children()[-1], values=avg_values)
        if loop:
            self.after(1000, self.update_loop)