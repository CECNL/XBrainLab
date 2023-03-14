from .base import Visiualizer
from matplotlib import pyplot as plt
import numpy as np

class SaliencyMapViz(Visiualizer):

    def get_plt(self, absolute):
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.clf()
        
        label_number = self.epoch_data.get_label_number()

        rows = 2
        cols = int(np.ceil(label_number / rows))
        labelIndex = 0
        for i in range(rows):
            for j in range(cols):
                if labelIndex >= label_number:
                    break
                plt.subplot(rows, cols, i * 2 + j + 1)
            
                saliency = self.get_gradient(labelIndex)
                if absolute:
                    saliency = np.abs(saliency).mean(axis=0)
                    if len(saliency) == 0:
                        continue
                    cmap='Reds'
                else:
                    if len(saliency) == 0:
                        continue
                    saliency = saliency.mean(axis=0)
                    cmap='bwr'
                im = plt.imshow(saliency, aspect='auto', cmap=cmap, 
                        vmin=saliency.min(), vmax=saliency.max(),  interpolation='none')
                plt.xlabel("sample")
                plt.ylabel("channel")
                plt.yticks(ticks=range(len(self.epoch_data.get_channel_names())), labels=self.epoch_data.get_channel_names(), fontsize=6)
                cbar = plt.colorbar(im, orientation='vertical')
                
                plt.title(f"Saliency Map of class {self.epoch_data.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return plt