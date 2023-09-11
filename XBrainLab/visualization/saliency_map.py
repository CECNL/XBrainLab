from .base import Visualizer
from matplotlib import pyplot as plt
import numpy as np

class SaliencyMapViz(Visualizer):
    """Visualizer that generate channel by time saliency map from evaluation record"""

    def _get_plt(self, absolute: bool) -> plt:
        """Return saliency map plot
        
        Args:
            absolute: whether to plot absolute value of saliency
            """
        label_number = self.epoch_data.get_label_number()
        # row and col of subplot
        duration = self.epoch_data.get_epoch_duration()
        rows = 1 if label_number <= 2 else 2
        cols = int(np.ceil(label_number / rows))
        # draw
        for labelIndex in range(label_number):
            plt.subplot(rows, cols, labelIndex + 1)
            saliency = self.get_gradient(labelIndex)
            # no test data for this label
            if len(saliency) == 0:
                continue

            if absolute:
                saliency = np.abs(saliency).mean(axis=0)
                cmap='Reds'
            else:
                saliency = saliency.mean(axis=0)
                cmap='coolwarm'
            
            im = plt.imshow(saliency, aspect='auto', cmap=cmap, 
                    vmin=saliency.min(), vmax=saliency.max(),  interpolation='none')
            
            plt.xlabel("time")
            plt.ylabel("channel")
            ch_names = self.epoch_data.get_channel_names()
            plt.yticks(ticks=range(len(ch_names)), labels=ch_names, fontsize=6)
            plt.xticks(ticks=np.linspace(0, saliency.shape[-1], 5), labels = np.round(np.linspace(0, duration, 5),2))
            plt.colorbar(im, orientation='vertical')
            plt.title(f"Saliency Map of class {self.epoch_data.label_map[labelIndex]}")
        plt.tight_layout()
        return plt