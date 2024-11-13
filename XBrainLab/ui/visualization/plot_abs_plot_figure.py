import tkinter as tk

from ..widget import PlotFigureWindow
from ...visualization import supported_saliency_methods

class PlotABSFigureWindow(PlotFigureWindow):
    def __init__(
        self,
        parent,
        trainers,
        plot_type,
        figsize=None,
        title='Plot',
        plan_name=None,
        real_plan_name=None,
        saliency_name=None,
        absolute=None
    ):
        super().__init__(
            parent, trainers, plot_type, figsize, title, plan_name, real_plan_name
        )
        self.saliency_name = saliency_name
        saliency_method_list = ['Select saliency method', 'Gradient', 'Gradient * Input', *supported_saliency_methods]
        ###+ select saliency method
        saliency_method_name = tk.StringVar(self)
        saliency_method_name.set(saliency_method_list[0])
        saliency_method_name.trace_add('write', self.on_saliency_method_select) # callback
        saliency_opt = tk.OptionMenu(self.selector_frame, saliency_method_name, *saliency_method_list) 
        saliency_opt.pack()

        self.saliency_opt = saliency_opt
        self.selected_saliency_method_name = saliency_method_name



        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace_add('write', self.absolute_callback)

        tk.Checkbutton(
            self.selector_frame, text='absolute value',
            var=self.absolute_var
        ).pack()

        if absolute is not None:
            self.absolute_var.set(absolute)
        if saliency_name:
            self.selected_saliency_method_name.set(saliency_name)

    def add_plot_command(self):
        if not hasattr(self, 'absolute_var'):
            return
        self.script_history.add_ui_cmd(
            "lab.show_grad_plot(plot_type="
            f"{self.plot_type.__class__.__name__}.{self.plot_type.name}, "
            f"plan_name={self.selected_plan_name.get()!r}, "
            f"real_plan_name={self.selected_real_plan_name.get()!r}, "
            f"saliency_name={self.selected_saliency_method_name.get()!r}, "
            f"absolute={self.absolute_var.get()!r})"
        )

    def on_saliency_method_select(self, var_name, *args):
        self.set_selection(False)
        if self.getvar(var_name) not in supported_saliency_methods and not self.getvar(var_name).startswith('Gradient'):
            return
        self.selected_saliency_method_name.set(self.getvar(var_name))
        self.add_plot_command()
        self.recreate_fig()


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
    def absolute_callback(self, *args):
        self.add_plot_command()
        self.recreate_fig()

    def _create_figure(self):
        if not hasattr(self, 'absolute_var'):
            return None

        eval_record = self.plan_to_plot.get_eval_record()
        if not eval_record:
            return None

        epoch_data = self.trainer.get_dataset().get_epoch_data()
        plot_visualizer = self.plot_type.value(
            eval_record, epoch_data, **self.get_figure_params()
        )
        figure = plot_visualizer.get_plt(method=self.selected_saliency_method_name.get(), absolute=self.absolute_var.get())
        return figure
