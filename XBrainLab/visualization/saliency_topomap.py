import mne
import numpy as np
from matplotlib import pyplot as plt

from .base import Visualizer


class SaliencyTopoMapViz(Visualizer):
    """Visualizer that generate topographic saliency map from evaluation record

    Args:
        absolute: whether to plot absolute value of saliency
    """
    def _get_plt(self, method, absolute: bool) -> plt:
        positions = self.epoch_data.get_montage_position()
        chs = self.epoch_data.get_channel_names()
        label_number = self.epoch_data.get_label_number()

        rows = 1 if label_number <= self.MIN_LABEL_NUMBER_FOR_MULTI_ROW else 2
        cols = int(np.ceil(label_number / rows))

        for labelIndex in range(label_number):
            ax = plt.subplot(rows, cols, labelIndex + 1)

            saliency = self.get_saliency(method, labelIndex)
            # no test data for this label
            if len(saliency) == 0:
                continue
            kwargs = {'pos': positions[:, 0:2],
                        'ch_type': 'eeg',
                        'sensors': False,
                        'names': chs,
                        'axes': ax,
                        'show': False,
                        'extrapolate': 'local',
                        'outlines': 'head',
                        'sphere': (0.0, -0.02, 0.0, 0.12),
                        }

            if absolute:
                saliency = np.abs(saliency).mean(axis=0)
                cmap = 'Reds'
            else:
                saliency = saliency.mean(axis=0)
                cmap = 'coolwarm'

            # average over time
            data = saliency.mean(axis=1)
            im, _ = mne.viz.plot_topomap(data=data, cmap=cmap, **kwargs)
            cbar = plt.colorbar(im, orientation='vertical')
            cbar.ax.get_yaxis().set_ticks([])
            plt.title(f"Saliency Map of class {self.epoch_data.label_map[labelIndex]}")
        plt.tight_layout()
        return plt
