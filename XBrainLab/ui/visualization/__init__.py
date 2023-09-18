from .montage_picker import PickMontageWindow

from .saliency_map import SaliencyMapWindow
from .saliency_topomap import SaliencyTopographicMapWindow
from .saliency_spectrogram import SaliencySpectrogramWindow
from .saliency_3Dplot import Saliency3DPlotWindow
from .plot_abs_plot_figure import PlotABSFigureWindow
from .plot_abs_topo_plot_figure import PlotTopoABSFigureWindow
from .model_summary import ModelSummaryWindow
from .plot_eval_record_figure import PlotEvalRecordFigureWindow

VISUALIZATION_MODULE_LIST = [
    SaliencyMapWindow, SaliencyTopographicMapWindow, 
    SaliencySpectrogramWindow,
    Saliency3DPlotWindow, ModelSummaryWindow
]

__all__ = [
    'VISUALIZATION_MODULE_LIST',
    'PickMontageWindow',
    'SaliencyMapWindow',
    'SaliencyTopographicMapWindow',
    'SaliencySpectrogramWindow',
    'Saliency3DPlotWindow',
    'PlotABSFigureWindow',
    'PlotTopoABSFigureWindow',
    'ModelSummaryWindow',
    'PlotEvalRecordFigureWindow'
]