import tkinter as tk
from ..base import TopWindow
from ..widget import PlotFigureWindow
import numpy as np
from matplotlib import pyplot as plt
from XBrainLab.utils import PlotType

class SaliencyMapWindow(PlotFigureWindow):
    command_label = 'Saliency map'
    def __init__(self, parent, trainers, plan_name=None, real_plan_name=None, absolute=None):
        super().__init__(parent, trainers, plot_type=PlotType.SALIENCY_MAP, title=self.command_label, plan_name=plan_name, real_plan_name=real_plan_name)
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace('w', self.absolute_callback)
        tk.Checkbutton(self.selector_frame, text='absolute value',
                                var=self.absolute_var).pack()
        if absolute is not None:
            self.absolute_var.set(absolute)

    def add_plot_command(self):
        if not hasattr(self, 'absolute_var'):
            return
        self.script_history.add_import("from XBrainLab.ui.visualization import SaliencyMapWindow")
        self.script_history.add_ui_cmd(f"study.show_grad_plot(plot_type=SaliencyMapWindow, plan_name={repr(self.selected_plan_name.get())}, real_plan_name={repr(self.selected_real_plan_name.get())}, absolute={repr(self.absolute_var.get())})")

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

        rows = 2
        cols = int(np.ceil(label_number / rows))
        plt.clf()
        labelIndex = 0
        
        for i in range(rows):
            for j in range(cols):
                if labelIndex >= label_number:
                    break
                plt.subplot(rows, cols, i * 2 + j + 1)
            
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
                im = plt.imshow(saliency, aspect='auto', cmap=cmap, 
                        vmin=saliency.min(), vmax=saliency.max(),  interpolation='none')
                plt.xlabel("sample")
                plt.ylabel("channel")
                plt.yticks(ticks=range(len(epoch_data.get_channel_names())), labels=epoch_data.get_channel_names(), fontsize=6)
                cbar = plt.colorbar(im, orientation='vertical')
                
                plt.title(f"Saliency Map of class {epoch_data.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return True