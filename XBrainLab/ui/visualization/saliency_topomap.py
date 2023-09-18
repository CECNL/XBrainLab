from .plot_abs_topo_plot_figure import PlotTopoABSFigureWindow
from XBrainLab.visualization import VisualizerType

class SaliencyTopographicMapWindow(PlotTopoABSFigureWindow):
    command_label = 'Saliency topographic map'
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
            plot_type=VisualizerType.SaliencyTopoMap, 
            title=self.command_label, 
            plan_name=plan_name, 
            real_plan_name=real_plan_name, 
            absolute=absolute
        )
    