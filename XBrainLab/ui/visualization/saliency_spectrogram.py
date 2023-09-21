from XBrainLab.visualization import VisualizerType

from .plot_eval_record_figure import PlotEvalRecordFigureWindow


class SaliencySpectrogramWindow(PlotEvalRecordFigureWindow):
    command_label = 'Saliency spectrogram'
    def __init__(
        self,
        parent,
        trainers,
        plan_name=None,
        real_plan_name=None
    ):
        super().__init__(
            parent,
            trainers,
            plot_type=VisualizerType.SaliencySpectrogramMap,
            title=self.command_label,
            plan_name=plan_name,
            real_plan_name=real_plan_name
        )
