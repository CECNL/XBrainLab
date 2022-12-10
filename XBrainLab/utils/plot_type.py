from enum import Enum

class PlotType(Enum):
    LOSS = 'get_loss_figure'
    ACCURACY = 'get_acc_figure'
    LR = 'get_lr_figure'
    CONFUSION = 'get_confusion_figure'
    SALIENCY_TOPOMAP = 'get_eval_record'
    SALIENCY_MAP = 'get_eval_record'
    