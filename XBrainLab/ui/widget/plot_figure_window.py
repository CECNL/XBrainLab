import tkinter as tk

from ..base import InitWindowValidateException
from ..script import Script
from .single_plot_window import SinglePlotWindow
from ...visualization import supported_saliency_methods


class PlotFigureWindow(SinglePlotWindow):
    def __init__(
        self,
        parent,
        trainers,
        plot_type,
        figsize=None,
        title='Plot',
        plan_name=None,
        real_plan_name=None,
        saliency_name=None
    ):
        super().__init__(parent, figsize, title=title)
        self.trainers = trainers
        self.trainer = None
        self.check_data()
        self.plot_type = plot_type
        self.plan_to_plot = None
        self.current_plot = None
        self.script_history = Script()
        self.script_history.add_import(
            "from XBrainLab.visualization import PlotType, VisualizerType"
        )
        self.plot_gap = 0
        self.saliency_name = saliency_name

        # init data
        ## fetch plan list
        trainer_map = {trainer.get_name(): trainer for trainer in trainers}
        trainer_list = ['Select a plan', *list(trainer_map.keys())]
        real_plan_list = ['Select repeat']
        saliency_method_list = ['Select saliency method', 'Gradient', 'Gradient * Input', *supported_saliency_methods]

        #+ gui
        ##+ option menu
        selector_frame = tk.Frame(self)
        ###+ plan
        selected_plan_name = tk.StringVar(self)
        selected_plan_name.set(trainer_list[0])
        selected_plan_name.trace_add('write', self.on_plan_select) # callback
        plan_opt = tk.OptionMenu(selector_frame, selected_plan_name, *trainer_list)
        ###+ real plan
        selected_real_plan_name = tk.StringVar(self)
        selected_real_plan_name.set(real_plan_list[0])
        selected_real_plan_name.trace_add('write', self.on_real_plan_select) # callback
        selected_plan_name.trace_add(
            'write', lambda *args, win=self: selected_real_plan_name.set(real_plan_list[0])
        )# reset selection
        real_plan_opt = tk.OptionMenu(
            selector_frame, selected_real_plan_name, *real_plan_list
        )
        ###+ select saliency method
        saliency_method_name = tk.StringVar(self)
        saliency_method_name.set(saliency_method_list[0])
        saliency_method_name.trace_add('write', self.on_saliency_method_select) # callback
        saliency_opt = tk.OptionMenu(selector_frame, saliency_method_name, *saliency_method_list) 

        plan_opt.pack()
        real_plan_opt.pack()
        saliency_opt.pack()
        selector_frame.grid(row=0, column=0, sticky='news')

        self.selector_frame = selector_frame
        self.plan_opt = plan_opt
        self.real_plan_opt = real_plan_opt
        self.saliency_opt = saliency_opt
        self.trainer_map = trainer_map
        self.real_plan_map = {}
        self.selected_plan_name = selected_plan_name
        self.selected_real_plan_name = selected_real_plan_name
        self.selected_saliency_method_name = saliency_method_name

        self.drawCounter = 0
        self.update_progress = -1
        self.update_loop()
        if plan_name:
            self.selected_plan_name.set(plan_name)
        if real_plan_name:
            self.selected_real_plan_name.set(real_plan_name)
        if saliency_name:
            self.selected_saliency_method_name.set(saliency_name)


    def check_data(self):
        if (
            not isinstance(self.trainers, list)
            or len(self.trainers) == 0
        ):
            raise InitWindowValidateException(
                self,
                'No valid training plan is generated'
            )

    def add_plot_command(self):
        self.script_history.add_ui_cmd(
            "lab.show_plot(plot_type="
            f"{self.plot_type.__class__.__name__}.{self.plot_type.name}, "
            f"plan_name={self.selected_plan_name.get()!r}, "
            f"real_plan_name={self.selected_real_plan_name.get()!r})"
            f"saliency_name={self.selected_saliency_method_name.get()!r}"
        )

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
            self.real_plan_opt['menu'].add_command(
                label=choice,
                command=lambda win=self, v=choice: win.selected_real_plan_name.set(v)
            )

    def on_real_plan_select(self, var_name, *args):
        self.set_selection(False)
        self.plan_to_plot = None
        if self.getvar(var_name) not in self.real_plan_map:
            return
        real_plan = self.real_plan_map[self.getvar(var_name)]
        self.plan_to_plot = real_plan
        self.add_plot_command()

    def on_saliency_method_select(self, var_name, *args):
        self.set_selection(False)
        if self.getvar(var_name) not in supported_saliency_methods and not self.getvar(var_name).startswith('Gradient'):
            return
        self.selected_saliency_method_name.set(self.getvar(var_name))
        self.add_plot_command()
        self.recreate_fig()
    
    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        figure = target_func(**self.get_figure_params())
        return figure

    def update_loop(self):
        if not self.window_exist:
            return
        counter = self.drawCounter
        MAX_PLOT_GAP = 20
        if self.current_plot != self.plan_to_plot:
            self.current_plot = self.plan_to_plot
            self.active_figure()
            if self.plan_to_plot is None:
                self.empty_data_figure()
            else:
                self.plot_gap += 1
                if self.plot_gap < MAX_PLOT_GAP:
                    self.current_plot = True
                else:
                    self.plot_gap = 0
                    update_progress = self.plan_to_plot.get_epoch()
                    if (update_progress != self.update_progress or
                        self.plan_to_plot.is_finished()
                    ):
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
            self.real_plan_map = {
                plan.get_name(): plan for plan in trainer.get_plans()
            }
            choice = trainer.get_plans()[item_count].get_name()
            self.real_plan_opt['menu'].add_command(
                label=choice,
                command=lambda win=self, v=choice: win.selected_real_plan_name.set(v)
            )
            item_count += 1

    def set_selection(self, allow):
        state = None
        if not allow:
            self.drawCounter += 1
            if self.plan_opt['state'] != tk.DISABLED:
                state = tk.DISABLED
        elif self.plan_opt['state'] == tk.DISABLED:
                state = tk.NORMAL
        if state:
            self.plan_opt.config(state=state)
            self.real_plan_opt.config(state=state)
            self.saliency_opt.config(state=state)

    def recreate_fig(self, *args, current_plot=True):
        self.update_progress = -1
        self.current_plot = current_plot
        self.plot_gap = 100

    def _get_script_history(self):
        return self.script_history
