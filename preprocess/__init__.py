from .preprocess import Channel, Filtering, Resample, EditEvent
from .epoching import TimeEpoch, WindowEpoch

PREPROCESS_MODULE_LIST = [Channel, Filtering, Resample, TimeEpoch, WindowEpoch, EditEvent]