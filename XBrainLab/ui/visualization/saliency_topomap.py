import tkinter as tk
from ..base import InitWindowValidateException
from ..widget import PlotFigureWindow
import numpy as np
from matplotlib import pyplot as plt
import mne
from XBrainLab.utils import PlotType

class SaliencyTopographicMapWindow(PlotFigureWindow):
    command_label = 'Saliency topographic map'
    def __init__(self, parent, trainers, plan_name=None, real_plan_name=None, absolute=None):
        super().__init__(parent, trainers, plot_type=PlotType.SALIENCY_MAP, title=self.command_label, plan_name=plan_name, real_plan_name=real_plan_name)
        self.check_dataset()
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace('w', self.absolute_callback)
        tk.Checkbutton(self.selector_frame, text='absolute value',
                                var=self.absolute_var).pack()
        if absolute is not None:
            self.absolute_var.set(absolute)

    def check_dataset(self):
        epoch_data = self.trainers[0].get_dataset().get_epoch_data()
        positions = epoch_data.get_montage_position()
        chs = epoch_data.get_channel_names()

        if positions is None:
            raise InitWindowValidateException(self, 'No valid montage position is set.')

    def add_plot_command(self):
        self.script_history.add_import("from XBrainLab.ui.visualization import SaliencyTopographicMapWindow")
        self.script_history.add_ui_cmd(f"study.show_grad_plot(plot_type=SaliencyTopographicMapWindow, plan_name={repr(self.selected_plan_name.get())}, real_plan_name={repr(self.selected_real_plan_name.get())}, absolute={repr(self.absolute_var.get())})")

    def absolute_callback(self, *args):
        self.add_plot_command()
        self.recreate_fig()

    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        eval_record = target_func()
        if not eval_record:
            return None
        
        epoch_data = self.trainer.get_dataset().get_epoch_data()
        label_number = epoch_data.get_label_number()

        positions = epoch_data.get_montage_position()
        chs = epoch_data.get_channel_names()

        rows = 2
        cols = int(np.ceil(label_number / rows))
        self.init_figure()
        labelIndex = 0
        
        for i in range(rows):
            for j in range(cols):
                if labelIndex >= label_number:
                    break
                ax = plt.subplot(rows, cols, i * cols + j + 1)
            
                if self.absolute_var.get():
                    saliency = np.abs(eval_record.gradient[labelIndex][eval_record.label == labelIndex]).mean(axis=0)
                    if len(saliency) == 0:
                        continue
                    cmap='Reds'
                else:
                    saliency = eval_record.gradient[labelIndex][eval_record.label == labelIndex]
                    if len(saliency) == 0:
                        continue
                    saliency = saliency.mean(axis=0)
                    cmap='bwr'
                
                data = saliency.mean(axis=1)
                im, _ = mne.viz.plot_topomap(data = data,
                                     pos = positions[:,0:2],
                                     show = False,
                                     names = chs,
                                     cmap=cmap,
                                     axes=ax,
                                     show_names=True)
                cbar = plt.colorbar(im, orientation='vertical')
                cbar.ax.get_yaxis().set_ticks([])
                plt.title(f"Saliency Map of class {epoch_data.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return True