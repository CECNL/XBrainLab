import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..dataset import Epochs
from ..training.record import EvalRecord


class Visualizer:
    """Base class for visualizer that generate figures from evaluation record

    Attributes:
        eval_record: evaluation record
        epoch_data: original epoch data for providing dataset information
        figsize: figure size
        dpi: figure dpi
        fig: figure to plot on. If None, a new figure will be created
    """
    MIN_LABEL_NUMBER_FOR_MULTI_ROW = 2
    def __init__(self,
                 eval_record: EvalRecord,
                 epoch_data: Epochs,
                 figsize: tuple = (6.4, 4.8),
                 dpi: int = 100,
                 fig: Figure = None):
        self.eval_record = eval_record
        self.epoch_data = epoch_data
        self.figsize = figsize
        self.dpi = dpi
        self.fig = fig

    def _get_plt(self, *args, **kwargs):
        raise NotImplementedError()

    def get_plt(self, *args, **kwargs):
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.clf()
        return self._get_plt(*args, **kwargs)


    def get_saliency(self, saliency_name, labelIndex: int) -> np.ndarray:
        """Return gradient of model by class index."""
        if saliency_name !=None:
            if saliency_name == "Gradient":
                return self.eval_record.get_gradient(labelIndex)
            elif saliency_name == "Gradient * Input":
                return self.eval_record.get_gradient_input(labelIndex)
            elif saliency_name == "SmoothGrad":
                return self.eval_record.get_smoothgrad(labelIndex)
            elif saliency_name == "SmoothGrad_Squared":
                return self.eval_record.get_smoothgrad_sq(labelIndex)
            elif saliency_name == "VarGrad":
                return self.eval_record.get_vargrad(labelIndex)
            else:
                raise NotImplementedError
