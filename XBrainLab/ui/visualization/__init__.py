from .model_summary import ModelSummaryWindow
from .montage_picker import PickMontageWindow
from .plot_abs_plot_figure import PlotABSFigureWindow
from .plot_abs_topo_plot_figure import PlotTopoABSFigureWindow
from .plot_eval_record_figure import PlotEvalRecordFigureWindow
from .saliency_setting import SetSaliencyWindow
from .saliency_3Dplot import Saliency3DPlotWindow
from .saliency_map import SaliencyMapWindow
from .saliency_spectrogram import SaliencySpectrogramWindow
from .saliency_topomap import SaliencyTopographicMapWindow
from .export_saliency import ExportSaliencyWindow

VISUALIZATION_MODULE_LIST = [
    SaliencyMapWindow, SaliencyTopographicMapWindow,
    SaliencySpectrogramWindow,
    Saliency3DPlotWindow, ModelSummaryWindow, ExportSaliencyWindow
]

__all__ = [
    'VISUALIZATION_MODULE_LIST',
    'PickMontageWindow',
    'SetSaliencyWindow',
    'SaliencyMapWindow',
    'SaliencyTopographicMapWindow',
    'SaliencySpectrogramWindow',
    'Saliency3DPlotWindow',
    'PlotABSFigureWindow',
    'PlotTopoABSFigureWindow',
    'ModelSummaryWindow',
    'PlotEvalRecordFigureWindow'
]
