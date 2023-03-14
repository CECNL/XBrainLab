import tkinter as tk

from ..base import InitWindowValidateException
from ..widget import PlotFigureWindow
from .plot_abs_plot_figure import PlotABSFigureWindow

class PlotTopoABSFigureWindow(PlotABSFigureWindow):
    def add_plot_command(self):
        if not hasattr(self, 'absolute_var'):
            return
        self.script_history.add_ui_cmd(f"study.show_grad_topo_plot(plot_type={self.plot_type.__name__}, plan_name={repr(self.selected_plan_name.get())}, real_plan_name={repr(self.selected_real_plan_name.get())}, absolute={repr(self.absolute_var.get())})")
    
    def check_data(self):
        super().check_data()
        epoch_data = self.trainers[0].get_dataset().get_epoch_data()
        positions = epoch_data.get_montage_position()
        chs = epoch_data.get_channel_names()

        if positions is None:
            raise InitWindowValidateException(self, 'No valid montage position is set.')