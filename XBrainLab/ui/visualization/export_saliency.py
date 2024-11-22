import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import pickle
import os
from ...visualization import supported_saliency_methods

from ..base import TopWindow

class ExportSaliencyWindow(TopWindow):
    command_label = "Export saliency (pickle)"
    def __init__(self, parent, trainers):
        super().__init__(parent, self.command_label)
        self.trainers = trainers

        self.trainer_map = {trainer.get_name(): trainer for trainer in trainers}
        trainer_list = ['---', *list(self.trainer_map.keys())]

        tk.Label(self, text="Select a plan: ").grid(row=3, column=0, sticky="w")
        self.selected_plan_name = ttk.Combobox(
            self, values=trainer_list, width=17, state="readonly"
        )
        self.selected_plan_name.grid(row=3, column=1, sticky="w")
        self.selected_plan_name.current(0)
        self.selected_plan_name.bind('<<ComboboxSelected>>', self._change_plan)

        tk.Label(self, text="Select a repeat: ").grid(row=4, column=0, sticky="w")
        self.selected_real_plan_name = ttk.Combobox(
            self, values=['---'], width=17, state="readonly"
        )
        self.selected_real_plan_name.grid(row=4, column=1, sticky="w")
        self.selected_real_plan_name.current(0)
        self.selected_real_plan_name.bind('<<ComboboxSelected>>', self._change_repeat)
        self.real_plan_opt = None

        tk.Label(self, text="Select a method: ").grid(row=5, column=0, sticky="w")
        self.selected_method_name = ttk.Combobox(
            self, values=['---'], width=17, state="readonly"
        )
        self.selected_method_name.grid(row=5, column=1, sticky="w")
        self.selected_method_name.current(0)

        tk.Button(self, text="Export location",
                  command=self._select_location, width=12).grid(row=6, columnspan=2)

    def _change_plan(self, event):
        if self.selected_plan_name.get() != '---':
            self.real_plan_opt = {
                plan.get_name(): plan
                for plan in self.trainer_map[self.selected_plan_name.get()].get_plans()
            }
            self.selected_real_plan_name['value'] = \
                ['---', *list(self.real_plan_opt.keys())]
        else:
            self.selected_real_plan_name['value'] = ['---']

    def _change_repeat(self, event):
        if self.selected_real_plan_name.get() != '---':
            real_plan = self.real_plan_opt[self.selected_real_plan_name.get()]
            self.selected_method_name['value'] = \
                ['---', 'Gradient', 'Gradient * Input', *supported_saliency_methods]
        else:
            self.selected_method_name['value'] = ['---']
    
    def _select_location(self):
        if (
                self.selected_plan_name.get() != '---'
                and self.selected_real_plan_name.get() != '---'
                and self.selected_method_name.get() != '---'
            ):
            real_plan = self.real_plan_opt[self.selected_real_plan_name.get()]
            eval_record = real_plan.get_eval_record()

            file_location = filedialog.askdirectory(title="Export Saliency")
            saliency = eval_record.export_saliency(self.selected_method_name.get(), file_location)
            if file_location:
                file_name = [self.selected_plan_name.get(), self.selected_real_plan_name.get(), self.selected_method_name.get()]
                with open(os.path.join(file_location, '_'.join(file_name)+'.pickle'), 'wb') as fp:
                    pickle.dump(saliency, fp, protocol=pickle.HIGHEST_PROTOCOL)

                self.destroy()
