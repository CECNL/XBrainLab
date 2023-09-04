from .base import Visualizer
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

class SaliencyMapViz(Visualizer):
    """Visualizer that generate channel by time saliency map from evaluation record"""

    def get_plt(self, absolute: bool, spectrogram: bool, sfreq: float) -> plt:
        """Return saliency map plot
        
        Args:
            absolute: whether to plot absolute value of saliency
            spectrogram: whether to plot as spectrogram
            sfreq: sampling frequency of the data
            """
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.clf()
        
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
            plt.subplot(rows, cols, i+1)
        
            saliency = self.get_gradient(labelIndex)
            # n, 22, 250
            if spectrogram:
                freqs, timestamps, saliency = signal.stft(saliency, fs=sfreq, axis=-1, nperseg=sfreq, noverlap=sfreq//2)
                saliency = np.mean(np.mean(abs(saliency), axis=0), axis=0) #[:saliency.shape[0]//2,:]
                cmap='coolwarm'
                im = plt.imshow(saliency, interpolation='gaussian', aspect='auto', cmap=cmap,
                                vmin=saliency.min(), vmax=saliency.max())
                tick_inteval = 0.5
                tick_label = np.round(np.arange(0, timestamps[-1], tick_inteval), 1)
                ticks = np.linspace(0, saliency.shape[1], len(tick_label))-tick_inteval
                plt.xlabel("time")
                plt.ylabel("frequency")
                plt.xticks(ticks=ticks, labels=tick_label, fontsize=6)
                plt.yticks(ticks=freqs[np.where(freqs%10==0)])
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
                im = plt.imshow(saliency, aspect='auto', cmap=cmap, 
                        vmin=saliency.min(), vmax=saliency.max(),  interpolation='none')
                plt.xlabel("sample")
                plt.ylabel("channel")
                plt.yticks(ticks=range(len(self.epoch_data.get_channel_names())), labels=self.epoch_data.get_channel_names(), fontsize=6)
            plt.colorbar(im, orientation='vertical')
            plt.title(f"Saliency Map of class {self.epoch_data.label_map[labelIndex]}")
            labelIndex += 1
        plt.tight_layout()
        return plt