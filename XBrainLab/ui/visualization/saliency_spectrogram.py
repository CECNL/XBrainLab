import tkinter as tk
from XBrainLab.visualization import VisualizerType

from .plot_eval_record_figure import PlotEvalRecordFigureWindow
from ...visualization import supported_saliency_methods


class SaliencySpectrogramWindow(PlotEvalRecordFigureWindow):
    command_label = 'Saliency spectrogram'
    def __init__(
        self,
        parent,
        trainers,
        plan_name=None,
        real_plan_name=None,
        saliency_name=None
    ):
        super().__init__(
            parent,
            trainers,
            plot_type=VisualizerType.SaliencySpectrogramMap,
            title=self.command_label,
            plan_name=plan_name,
            real_plan_name=real_plan_name
        )
        saliency_method_list = ['Select saliency method', 'Gradient', 'Gradient * Input', *supported_saliency_methods]
        ###+ select saliency method
        saliency_method_name = tk.StringVar(self)
        saliency_method_name.set(saliency_method_list[0])
        saliency_method_name.trace_add('write', self.on_saliency_method_select) # callback
        saliency_opt = tk.OptionMenu(self.selector_frame, saliency_method_name, *saliency_method_list) 
        saliency_opt.pack()

        self.saliency_opt = saliency_opt
        self.selected_saliency_method_name = saliency_method_name

    def on_saliency_method_select(self, var_name, *args):
        self.set_selection(False)
        if self.getvar(var_name) not in supported_saliency_methods and not self.getvar(var_name).startswith('Gradient'):
            return
        self.selected_saliency_method_name.set(self.getvar(var_name))
        self.add_plot_command()
        self.recreate_fig()
