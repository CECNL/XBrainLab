import os
import torch
import numpy as np
import mne
from matplotlib import pyplot as plt

def calculate_confusion(output, label):
    classNum = len(np.unique(label))
    confusion = np.zeros((classNum,classNum), dtype=np.uint32)
    output = output.argmax(axis=1)
    for ground_truth in range(classNum):
        for predict in range(classNum):
            confusion[ground_truth][predict] = (output[label == ground_truth] == predict).sum()
    return confusion


class EvalRecord:
    def __init__(self, label, output, gradient):
        self.label = label
        self.output = output
        self.gradient = gradient
    
    def export(self, target_path):
        record = {
            'label': self.label,
            'output': self.output,
            'gradient': self.gradient,
        }
        torch.save(record, os.path.join(target_path, 'eval'))
    
    def export_csv(self, target_path):
        data = np.c_[self.output, self.label, self.output.argmax(axis=1)]
        np.savetxt(target_path, data, delimiter=',', newline='\n', header=f'{",".join([str(i) for i in range(self.output.shape[1])])},ground_truth,predict', comments='')
    #
    def get_acc(self):
        return sum(self.output.argmax(axis=1) == self.label) / len(self.label)

    def get_kappa(self):
        confusion = calculate_confusion(self.output, self.label)
        classNum = len(confusion)
        P0 = np.diagonal(confusion).sum() / confusion.sum()
        Pe = sum([confusion[:,i].sum() * confusion[i].sum() for i in range(classNum)]) / (confusion.sum() * confusion.sum())
        return (P0 - Pe) / (1 - Pe)
    # figure
    def get_saliency_map(self, epoch_data, absolute, fig=None, figsize=(6.4, 4.8), dpi=100):
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        label_number = epoch_data.get_label_number()

        rows = 2
        cols = int(np.ceil(label_number / rows))
        labelIndex = 0
        for i in range(rows):
            for j in range(cols):
                if labelIndex >= label_number:
                    break
                plt.subplot(rows, cols, i * 2 + j + 1)
            
                if absolute:
                    saliency = np.abs(self.gradient[labelIndex][self.label == labelIndex]).mean(axis=0)
                    if len(saliency) == 0:
                        continue
                    cmap='Reds'
                else:
                    saliency = self.gradient[labelIndex][self.label == labelIndex]
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
        return plt

    def get_saliency_topo_map(self, epoch_data, absolute, fig=None, figsize=(6.4, 4.8), dpi=100):
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        positions = epoch_data.get_montage_position()
        chs = epoch_data.get_channel_names()
        label_number = epoch_data.get_label_number()

        rows = 2
        cols = int(np.ceil(label_number / rows))
        labelIndex = 0
        
        for i in range(rows):
            for j in range(cols):
                if labelIndex >= label_number:
                    break
                ax = plt.subplot(rows, cols, i * cols + j + 1)
            
                if absolute:
                    saliency = np.abs(self.gradient[labelIndex][self.label == labelIndex]).mean(axis=0)
                    if len(saliency) == 0:
                        continue
                    cmap='Reds'
                else:
                    saliency = self.gradient[labelIndex][self.label == labelIndex]
                    if len(saliency) == 0:
                        continue
                    saliency = saliency.mean(axis=0)
                    cmap='bwr'
                
                data = saliency.mean(axis=1)
                im, _ = mne.viz.plot_topomap(data = data,
                                     pos = positions[:,0:2],
                                     names = chs,
                                     cmap=cmap,
                                     axes=ax,
                                     show_names=True,
                                     show=False)
                cbar = plt.colorbar(im, orientation='vertical')
                cbar.ax.get_yaxis().set_ticks([])
                plt.title(f"Saliency Map of class {epoch_data.label_map[labelIndex]}")
                labelIndex += 1
        plt.tight_layout()
        return plt
    