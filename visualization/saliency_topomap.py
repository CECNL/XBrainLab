import tkinter as tk
from ..widget import PlotFigureWindow, PlotType
import numpy as np
from matplotlib import pyplot as plt
import mne

class SaliencyTopographicMapWindow(PlotFigureWindow):
    command_label = 'Saliency topographic map'
    def __init__(self, parent, trainers):
        super().__init__(parent, trainers, plot_type=PlotType.SALIENCY_MAP, title=self.command_label)
        if not self.is_valid():
            return
        if not self.check_dataset():
            return
        self.absolute_var = tk.BooleanVar(self)
        self.absolute_var.trace('w', self.absolute_callback)
        tk.Checkbutton(self.selector_frame, text='absolute value',
                                var=self.absolute_var).pack()
    
    def check_dataset(self):
        data_holder = self.trainers[0].get_dataset().get_data_holder()
        positions = data_holder.get_montage_position()
        chs = data_holder.get_channel_names()

        if positions is None:
            self.valid = False
            self.withdraw()
            tk.messagebox.showerror(parent=self.master, title='Error', message='No valid montage position is set.')
            self.destroy()
            return False
        return True

    def absolute_callback(self, *args):
        self.current_plot = True
        self.plot_gap = 100

    def _create_figure(self):
        target_func = getattr(self.plan_to_plot, self.plot_type.value)
        eval_record = target_func()
        if not eval_record:
            return None
        
        data_holder = self.trainer.get_dataset().get_data_holder()
        label_number = data_holder.get_label_number()

        positions = data_holder.get_montage_position()
        chs = data_holder.get_channel_names()

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
                plt.title(f"Saliency Map of class {data_holder.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return True