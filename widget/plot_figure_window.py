import tkinter as tk
from tkinter import messagebox
from enum import Enum
from . import SinglePlotWindow

class PlotType(Enum):
    LOSS = 'get_loss_figure'
    ACCURACY = 'get_acc_figure'
    LR = 'get_lr_figure'
    CONFUSION = 'get_confusion_figure'
    SALIENCY_TOPOMAP = 'get_eval_record'
    SALIENCY_MAP = 'get_eval_record'
    
class PlotFigureWindow(SinglePlotWindow):
    def __init__(self, parent, training_plan_holders, plot_type, figsize=None, title='Plot'):
        super().__init__(parent, figsize, title=title)
        self.training_plan_holders = training_plan_holders
        self.plan_holder = None
        if not self.check_data():
            return
        self.plot_type = plot_type
        self.plan_to_plot = None
        self.current_plot = None
        self.plot_gap = 0

        # init data
        ## fetch plan list
        training_plan_map = {plan_holder.get_name(): plan_holder for plan_holder in training_plan_holders}
        training_plan_list = ['Select a plan'] + list(training_plan_map.keys())
        real_plan_list = ['Select repeat']

        #+ gui
        ##+ option menu
        selector_frame = tk.Frame(self)
        ###+ plan
        selected_plan_name = tk.StringVar(self)
        selected_plan_name.set(training_plan_list[0])
        selected_plan_name.trace('w', self.on_plan_select) # callback
        plan_opt = tk.OptionMenu(selector_frame, selected_plan_name, *training_plan_list)
        ###+ real plan
        selected_real_plan_name = tk.StringVar(self)
        selected_real_plan_name.set(real_plan_list[0])
        selected_real_plan_name.trace('w', self.on_real_plan_select) # callback
        selected_plan_name.trace('w', lambda *args: selected_real_plan_name.set(real_plan_list[0])) # reset selection
        real_plan_opt = tk.OptionMenu(selector_frame, selected_real_plan_name, *real_plan_list)

        plan_opt.pack()
        real_plan_opt.pack()
        selector_frame.grid(row=0, column=0, sticky='news')

        self.selector_frame = selector_frame
        self.plan_opt = plan_opt
        self.real_plan_opt = real_plan_opt
        self.training_plan_map = training_plan_map
        self.real_plan_map = {}
        self.selected_plan_name = selected_plan_name
        self.selected_real_plan_name = selected_real_plan_name

        self.drawCounter = 0
        self.update_loop()

    def check_data(self):
        if type(self.training_plan_holders) != list or len(self.training_plan_holders) == 0:
            self.withdraw()
            tk.messagebox.showerror(parent=self, title='Error', message='No valid training plan is generated')
            self.destroy()
            return False
        return True

    def on_plan_select(self, var_name, *args):
        self.set_selection(False)
        self.plan_to_plot = None
        self.plan_holder = None
        item_count = self.real_plan_opt['menu'].index(tk.END)
        if item_count >= 1:
            self.real_plan_opt['menu'].delete(1, item_count)
        if self.getvar(var_name) not in self.training_plan_map:
            return
        plan_holder = self.training_plan_map[self.getvar(var_name)]
        if plan_holder is None:
            return
        self.plan_holder = plan_holder
        self.real_plan_map = {plan.get_name(): plan for plan in plan_holder.get_plans()}
        for choice in self.real_plan_map:
            self.real_plan_opt['menu'].add_command(label=choice, command=lambda value=choice: self.selected_real_plan_name.set(value))

    def on_real_plan_select(self, var_name, *args):
        self.set_selection(False)
        self.plan_to_plot = None
        if self.getvar(var_name) not in self.real_plan_map:
            return
        real_plan = self.real_plan_map[self.getvar(var_name)]
        self.plan_to_plot = real_plan
        self.plot_gap = 100

    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        figure = target_func(**self.get_figure_parms())
        return figure

    def update_loop(self):
        if self.winfo_exists() == 0:
            return
        counter = self.drawCounter
        if self.current_plot != self.plan_to_plot:
            self.current_plot = self.plan_to_plot
            self.active_figure()
            if self.plan_to_plot is None:
                self.empty_data_figure()
            else:
                self.plot_gap += 1
                if self.plot_gap < 20:
                    self.current_plot = True
                else:
                    self.plot_gap = 0
                    self.show_drawing()
                    figure = self._create_figure()
                    if figure is None:
                        self.empty_data_figure()
                    if not self.plan_to_plot.is_finished():
                        self.current_plot = True
                    self.redraw()
        if counter == self.drawCounter:
            self.set_selection(allow=True)

        self.after(100, self.update_loop)

        item_count = self.real_plan_opt['menu'].index(tk.END)
        if self.selected_plan_name.get() not in self.training_plan_map:
            return
        plan_holder = self.training_plan_map[self.selected_plan_name.get()]
        while len(plan_holder.get_plans()) > 0 and item_count < len(plan_holder.get_plans()):
            self.real_plan_map = {plan.get_name(): plan for plan in plan_holder.get_plans()}
            choice = plan_holder.get_plans()[item_count].get_name()
            self.real_plan_opt['menu'].add_command(label=choice, command=lambda value=choice: self.selected_real_plan_name.set(value))
            item_count += 1

    def set_selection(self, allow):
        state = None
        if not allow:
            self.drawCounter += 1
            if self.plan_opt['state'] != tk.DISABLED:
                state = tk.DISABLED
        else:
            if self.plan_opt['state'] == tk.DISABLED:
                state = tk.NORMAL
        if state:
            self.plan_opt.config(state=state)
            self.real_plan_opt.config(state=state)
        