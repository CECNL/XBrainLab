from ..base import InitWindowValidateException
from .plot_abs_plot_figure import PlotABSFigureWindow


class PlotTopoABSFigureWindow(PlotABSFigureWindow):
    def add_plot_command(self):
        if not hasattr(self, 'absolute_var'):
            return
        self.script_history.add_ui_cmd(
            "lab.show_grad_topo_plot(plot_type="
            f"{self.plot_type.__class__.__name__}.{self.plot_type.name}, "
            f"plan_name={self.selected_plan_name.get()!r}, "
            f"real_plan_name={self.selected_real_plan_name.get()!r}, "
            f"absolute={self.absolute_var.get()!r})"
        )

    def check_data(self):
        super().check_data()
        epoch_data = self.trainers[0].get_dataset().get_epoch_data()
        positions = epoch_data.get_montage_position()

        if positions is None:
            raise InitWindowValidateException(
                self, 'No valid montage position is set.'
            )
