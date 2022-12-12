import tkinter as tk
from ..widget import PlotFigureWindow

from XBrainLab.visualization import PlotType

class ConfusionMatrixWindow(PlotFigureWindow):
    command_label = 'Confusion matrix'
    def __init__(self, parent, trainers):
        super().__init__(parent, trainers, plot_type=PlotType.CONFUSION)