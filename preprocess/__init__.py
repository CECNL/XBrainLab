from .channel_selection import ChannelSelection
from .filtering import Filtering
from .resample import Resample
from .time_epoch import TimeEpoch
from .window_epoch import WindowEpoch
from .edit_event import EditEvent

PREPROCESS_MODULE_LIST = [ChannelSelection, Filtering, Resample, TimeEpoch, WindowEpoch, EditEvent]