from enum import Enum
from .saliency_map import SaliencyMapViz
from .saliency_topomap import SaliencyTopoMapViz
from .saliency_spectrogram_map import SaliencySpectrogramMapViz
class PlotType(Enum):
    """Utility class for type of training plot."""
    LOSS = 'get_loss_figure'
    ACCURACY = 'get_acc_figure'
    AUC = 'get_auc_figure'
    LR = 'get_lr_figure'
    CONFUSION = 'get_confusion_figure'
    
class VisualizerType(Enum):
    SaliencyMap = SaliencyMapViz
    SaliencyTopoMap = SaliencyTopoMapViz
    SaliencySpectrogramMap = SaliencySpectrogramMapViz
