import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt

from ..base import TopWindow
from ..widget import PlotFigureWindow
from .plot_abs_plot_figure import PlotABSFigureWindow
from XBrainLab.visualization import SaliencyMapViz

class SaliencyMapWindow(PlotABSFigureWindow):
    command_label = 'Saliency map'
    def __init__(self, parent, trainers, plan_name=None, real_plan_name=None, absolute=None, spectrogram=None):
        super().__init__(parent, trainers, plot_type=SaliencyMapViz, title=self.command_label, plan_name=plan_name, real_plan_name=real_plan_name, absolute=absolute, spectrogram=spectrogram)
