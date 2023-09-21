from XBrainLab.visualization import PlotType

from ..widget import PlotFigureWindow


class ConfusionMatrixWindow(PlotFigureWindow):
    command_label = 'Confusion matrix'
    def __init__(self, parent, trainers):
        super().__init__(parent, trainers, plot_type=PlotType.CONFUSION)
