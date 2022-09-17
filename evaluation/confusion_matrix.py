import tkinter as tk
from ..widget import PlotFigureWindow, PlotType

class ConfusionMatrixWindow(PlotFigureWindow):
    command_label = 'Confusion matrix'
    def __init__(self, parent, training_plan_holders):
        super().__init__(parent, training_plan_holders, plot_type=PlotType.CONFUSION)