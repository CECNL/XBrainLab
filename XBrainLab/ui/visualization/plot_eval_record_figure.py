from ..widget import PlotFigureWindow


class PlotEvalRecordFigureWindow(PlotFigureWindow):

    def add_plot_command(self):
        self.script_history.add_ui_cmd(
            "lab.show_grad_eval_plot(plot_type="
            f"{self.plot_type.__class__.__name__}.{self.plot_type.name}, "
            f"plan_name={self.selected_plan_name.get()!r}, "
            f"real_plan_name={self.selected_real_plan_name.get()!r})"
        )

    def _create_figure(self):
        eval_record = self.plan_to_plot.get_eval_record()
        if not eval_record:
            return None

        epoch_data = self.trainer.get_dataset().get_epoch_data()
        plot_visualizer = self.plot_type.value(
            eval_record, epoch_data, **self.get_figure_params()
        )
        figure = plot_visualizer.get_plt()
        return figure
