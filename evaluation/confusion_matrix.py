import tkinter as tk
from ..training.training_plot import TrainingPlotWindow, TrainingPlotType

class ConfusionMatrixWindow(TrainingPlotWindow):
    command_label = 'Confusion matrix'
    def __init__(self, parent, training_plan_holders):
        super().__init__(parent, training_plan_holders, plot_type=TrainingPlotType.CONFUSION)