import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt

from ..base import TopWindow
from ..widget import PlotFigureWindow
from XBrainLab.visualization import PlotType
from .plot_3d_head import Saliency3D


class Saliency3DPlotWindow(TopWindow):
    command_label = '3D Saliency plot'
    def __init__(self, parent, trainers, plan_name=None, real_plan_name=None, absolute=None):
        super().__init__(parent, self.command_label)
        self.trainers = trainers

        self.trainer_map = {trainer.get_name(): trainer for trainer in trainers}
        trainer_list = ['---'] + list(self.trainer_map.keys())

        tk.Label(self, text="Select a plan: ").grid(row=3, column=0, sticky="w")
        self.selected_plan_name = ttk.Combobox(self, values=trainer_list, width=17, state="readonly")
        self.selected_plan_name.grid(row=3, column=1, sticky="w")
        self.selected_plan_name.current(0)
        self.selected_plan_name.bind('<<ComboboxSelected>>', self._change_plan)

        tk.Label(self, text="Select a repeat: ").grid(row=4, column=0, sticky="w")
        self.selected_real_plan_name = ttk.Combobox(self, values=['---'], width=17, state="readonly")
        self.selected_real_plan_name.grid(row=4, column=1, sticky="w")
        self.selected_real_plan_name.current(0)
        self.selected_real_plan_name.bind('<<ComboboxSelected>>', self._change_repeat)
        self.real_plan_opt = None

        tk.Label(self, text="Select a event: ").grid(row=5, column=0, sticky="w")
        self.selected_event_name = ttk.Combobox(self, values=['---'], width=17, state="readonly")
        self.selected_event_name.grid(row=5, column=1, sticky="w")
        self.selected_event_name.current(0)

        tk.Button(self, text="Confirm", command=self._show_plot, width=8).grid(row=6, columnspan=2)

    def _change_plan(self, event):
        if self.selected_plan_name.get() != '---':
            self.real_plan_opt = {plan.get_name(): plan for plan in self.trainer_map[self.selected_plan_name.get()].get_plans()}
            self.selected_real_plan_name['value'] = ['---'] + list(self.real_plan_opt.keys())
        else:
            self.selected_real_plan_name['value'] = ['---']

    def _change_repeat(self, event):
        if self.selected_real_plan_name.get() != '---':
            real_plan = self.real_plan_opt[self.selected_real_plan_name.get()]
            self.selected_event_name['value'] = ['---'] + list(real_plan.dataset.get_epoch_data().event_id.keys())
        else:
            self.selected_event_name['value'] = ['---']

    def _show_plot(self):
        if self.selected_plan_name.get() != '---' and  self.selected_real_plan_name.get() != '---' and self.selected_event_name.get() != '---':
            real_plan = self.real_plan_opt[self.selected_real_plan_name.get()]
            events = real_plan.dataset.get_epoch_data().event_id
            eval_record = real_plan.get_eval_record()
            epoch_data = real_plan.dataset.get_epoch_data()

            # draw 3d plot
            saliency = Saliency3D(eval_record, epoch_data, self.selected_event_name.get())
            plot = saliency.get3dHeadPlot()
            plot.show()
