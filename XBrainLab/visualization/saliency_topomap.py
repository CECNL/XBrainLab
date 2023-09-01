from .base import Visualizer
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import mne

class SaliencyTopoMapViz(Visualizer):

    def get_plt(self, absolute, spectrogram, sfreq):
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.clf()

        positions = self.epoch_data.get_montage_position()
        chs = self.epoch_data.get_channel_names()
        label_number = self.epoch_data.get_label_number()

        if label_number<=2:
            rows = 1
        else:
            rows = 2
        cols = int(np.ceil(label_number / rows))
        labelIndex = 0
        
        for i in range(label_number):
            if labelIndex >= label_number:
                break
            ax = plt.subplot(rows, cols, i+1)
        
            saliency = self.get_gradient(labelIndex)
            kwargs = dict(pos = positions[:,0:2],
                        ch_type = 'eeg',
                        sensors = False,
                        names = chs,
                        axes=ax,
                        show=False,
                        extrapolate='local',
                        outlines='head',
                        sphere=(0.0, -0.02, 0.0, 0.12),
                        )

            if spectrogram:
                reqs, timestamps, saliency = signal.stft(saliency, fs=sfreq, axis=-1, nperseg=sfreq, noverlap=sfreq//2)
                # saliency = np.mean(np.mean(abs(saliency**2), axis=0), axis=0) #  trial, C, freq band, time interval -> freq, time interval
                cmap='coolwarm'
                saliency = np.mean(np.mean(np.mean(abs(saliency), axis=0), axis=-1), axis=-1)
                im, _ = mne.viz.plot_topomap(data = saliency, cmap=cmap, **kwargs)
            else:
                if absolute:
                    saliency = np.abs(saliency).mean(axis=0)
                    if len(saliency) == 0:
                        continue
                    cmap='Reds'
                else:
                    if len(saliency) == 0:
                        continue
                    saliency = saliency.mean(axis=0)
                    cmap='coolwarm'
                
                data = saliency.mean(axis=1)
                im, _ = mne.viz.plot_topomap(data = data, cmap=cmap, **kwargs)
            cbar = plt.colorbar(im, orientation='vertical')
            cbar.ax.get_yaxis().set_ticks([])
            plt.title(f"Saliency Map of class {self.epoch_data.label_map[labelIndex]}")
            labelIndex += 1
        plt.tight_layout()
        return plt
    