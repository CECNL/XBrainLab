import tkinter as tk

from ..widget import PlotFigureWindow


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
            parent, trainers, plot_type, figsize, title, plan_name, real_plan_name, saliency_name
        )
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace_add('write', self.absolute_callback)

        tk.Checkbutton(
            self.selector_frame, text='absolute value',
            var=self.absolute_var
        ).pack()

        if absolute is not None:
            self.absolute_var.set(absolute)

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
