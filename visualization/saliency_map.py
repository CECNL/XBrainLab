import tkinter as tk
from ..base import TopWindow
from ..widget import PlotFigureWindow, PlotType
import numpy as np
from matplotlib import pyplot as plt

class SaliencyMapWindow(PlotFigureWindow):
    command_label = 'Saliency map'
    def __init__(self, parent, training_plan_holders):
        super().__init__(parent, training_plan_holders, plot_type=PlotType.SALIENCY_MAP, title=self.command_label)
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace('w', self.absolute_callback)
        tk.Checkbutton(self.selector_frame, text='absolute value',
                                var=self.absolute_var).pack()

    def absolute_callback(self, *args):
        self.current_plot = True
        self.plot_gap = 100

    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        eval_record = target_func()
        if not eval_record:
            return None
        
        data_holder = self.plan_holder.get_dataset().get_data_holder()
        label_number = data_holder.get_label_number()

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
                plt.yticks(ticks=range(len(data_holder.get_channel_names())), labels=data_holder.get_channel_names(), fontsize=6)
                cbar = plt.colorbar(im, orientation='vertical')
                
                plt.title(f"Saliency Map of class {data_holder.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return True