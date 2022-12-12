import tkinter as tk
from . import SinglePlotWindow
from ..base import InitWindowValidateException
from ..script import Script
from XBrainLab.visualization import PlotType

class PlotFigureWindow(SinglePlotWindow):
    def __init__(self, parent, trainers, plot_type, figsize=None, title='Plot', plan_name=None, real_plan_name=None):
        super().__init__(parent, figsize, title=title)
        self.trainers = trainers
        self.trainer = None
        self.check_data()
        self.plot_type = plot_type
        self.plan_to_plot = None
        self.current_plot = None
        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.visualization import PlotType")
        self.plot_gap = 0

        # init data
        ## fetch plan list
        trainer_map = {trainer.get_name(): trainer for trainer in trainers}
        trainer_list = ['Select a plan'] + list(trainer_map.keys())
        real_plan_list = ['Select repeat']

        #+ gui
        ##+ option menu
        selector_frame = tk.Frame(self)
        ###+ plan
        selected_plan_name = tk.StringVar(self)
        selected_plan_name.set(trainer_list[0])
        selected_plan_name.trace('w', self.on_plan_select) # callback
        plan_opt = tk.OptionMenu(selector_frame, selected_plan_name, *trainer_list)
        ###+ real plan
        selected_real_plan_name = tk.StringVar(self)
        selected_real_plan_name.set(real_plan_list[0])
        selected_real_plan_name.trace('w', self.on_real_plan_select) # callback
        selected_plan_name.trace('w', lambda *args,win=self: selected_real_plan_name.set(real_plan_list[0])) # reset selection
        real_plan_opt = tk.OptionMenu(selector_frame, selected_real_plan_name, *real_plan_list)

        plan_opt.pack()
        real_plan_opt.pack()
        selector_frame.grid(row=0, column=0, sticky='news')

        self.selector_frame = selector_frame
        self.plan_opt = plan_opt
        self.real_plan_opt = real_plan_opt
        self.trainer_map = trainer_map
        self.real_plan_map = {}
        self.selected_plan_name = selected_plan_name
        self.selected_real_plan_name = selected_real_plan_name
        
        self.drawCounter = 0
        self.update_progress = -1
        self.update_loop()
        if plan_name:
            self.selected_plan_name.set(plan_name)
        if real_plan_name:
            self.selected_real_plan_name.set(real_plan_name)


    def check_data(self):
        if type(self.trainers) != list or len(self.trainers) == 0:
            raise InitWindowValidateException(self, 'No valid training plan is generated')

    def add_plot_command(self):
        self.script_history.add_ui_cmd(f"study.show_plot(plot_type={self.plot_type.__class__.__name__}.{self.plot_type.name}, plan_name={repr(self.selected_plan_name.get())}, real_plan_name={repr(self.selected_real_plan_name.get())})")

    def on_plan_select(self, var_name, *args):
        self.set_selection(False)
        self.plan_to_plot = None
        self.trainer = None
        item_count = self.real_plan_opt['menu'].index(tk.END)
        if item_count >= 1:
            self.real_plan_opt['menu'].delete(1, item_count)
        if self.getvar(var_name) not in self.trainer_map:
            return
        trainer = self.trainer_map[self.getvar(var_name)]
        if trainer is None:
            return
        self.trainer = trainer
        self.real_plan_map = {plan.get_name(): plan for plan in trainer.get_plans()}
        for choice in self.real_plan_map:
            self.real_plan_opt['menu'].add_command(label=choice, command=lambda win=self,value=choice: self.selected_real_plan_name.set(value))

    def on_real_plan_select(self, var_name, *args):
        self.set_selection(False)
        self.plan_to_plot = None
        if self.getvar(var_name) not in self.real_plan_map:
            return
        real_plan = self.real_plan_map[self.getvar(var_name)]
        self.plan_to_plot = real_plan
        self.add_plot_command()
        self.recreate_fig()

    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        figure = target_func(**self.get_figure_parms())
        return figure

    def update_loop(self):
        if not self.window_exist:
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
                    update_progress = self.plan_to_plot.get_epoch()
                    if update_progress != self.update_progress or self.plan_to_plot.is_finished():
                        self.update_progress = update_progress
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
        if self.selected_plan_name.get() not in self.trainer_map:
            return
        trainer = self.trainer_map[self.selected_plan_name.get()]
        while len(trainer.get_plans()) > 0 and item_count < len(trainer.get_plans()):
            self.real_plan_map = {plan.get_name(): plan for plan in trainer.get_plans()}
            choice = trainer.get_plans()[item_count].get_name()
            self.real_plan_opt['menu'].add_command(label=choice, command=lambda win=self,value=choice: self.selected_real_plan_name.set(value))
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
    
    def recreate_fig(self, *args, current_plot=True):
        self.update_progress = -1
        self.current_plot = current_plot
        self.plot_gap = 100

    def _get_script_history(self):
        return self.script_history