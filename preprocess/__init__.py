from .channel import Channel
from .filtering import Filtering
from .resample import Resample
from .timeEpoch import TimeEpoch
from .windowEpoch import WindowEpoch
from .editEvent import EditEvent

PREPROCESS_MODULE_LIST = [Channel, Filtering, Resample, TimeEpoch, WindowEpoch, EditEvent]