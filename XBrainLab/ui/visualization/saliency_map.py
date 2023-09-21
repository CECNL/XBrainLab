from XBrainLab.visualization import VisualizerType

from .plot_abs_plot_figure import PlotABSFigureWindow


class SaliencyMapWindow(PlotABSFigureWindow):
    command_label = 'Saliency map'
    def __init__(
        self,
        parent,
        trainers,
        plan_name=None,
        real_plan_name=None,
        absolute=None
    ):
        super().__init__(
            parent,
            trainers,
            plot_type=VisualizerType.SaliencyMap,
            title=self.command_label,
            plan_name=plan_name,
            real_plan_name=real_plan_name,
            absolute=absolute
        )
