import numpy as np
import mne
from .base import DataType
from ...base import TopWindow
from ...dataset.data_holder import Raw

class RawInfo:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sfreq = None
        self.nchan = None
        self.ntimes = None
        self.tmin = None
        self.channel_info = None
        self.channel_type = None

    def is_info_complete(self):
        if not self.sfreq:
            return False
        return True

    def set_attr(self, sfreq, nchan, ntimes, tmin):
        self.sfreq = sfreq
        self.nchan = nchan
        self.ntimes = ntimes
        self.tmin = tmin

    def set_shape_idx(self, channel_info, channel_type):
        self.channel_info = channel_info
        self.channel_type = channel_type

    def get_sfreq(self):
        return self.sfreq

    def get_tmin(self):
        return self.tmin

    def reshape_array(self, data):
        reshape_idx = []
        if len(data.shape) > 2:
            epoch_idx = self.channel_info.index(self.channel_type.EPOCH)
            reshape_idx = [epoch_idx]
        ch_idx = self.channel_info.index(self.channel_type.CH)
        time_idx = self.channel_info.index(self.channel_type.TIME)
        reshape_idx += [ch_idx, time_idx]
        data = np.transpose(data, reshape_idx)
        return data

    def generate_mne(self, filepath, data_array, data_type):
        data_array = self.reshape_array(data_array)
        data_info = mne.create_info(self.nchan, self.sfreq, 'eeg')
        
        if data_type == DataType.RAW.value:
            mne_data = mne.io.RawArray(data_array, data_info)
        elif data_type == DataType.EPOCH.value:
            mne_data = mne.EpochsArray(data = data_array, info=data_info, tmin=self.tmin)
        return mne_data
   
class DictInfo(RawInfo):
    def __init__(self):
        super().__init__()
        self.data_keys = set()
        self.event_keys = set()

    def reset(self):
        super().reset()
        self.reset_keys()

    def reset_keys(self):
        self.data_keys = set()
        self.event_keys = set()

    def is_info_complete(self, selected_data):
        if not super().is_info_complete():
            return False
        if not self.data_keys.intersection(selected_data.keys()):
            return False
        return True
    
    def add_keys(self, data, event):
        self.data_keys.add(data)
        if event:
            self.event_keys.add(event)

    def generate_mne(self, filepath, selected_data, data_type):
        data_array = event_array = None
        for k in self.data_keys:
            if k in selected_data.keys():
                data_array = selected_data[k]
                break
        if data_array is None:
            raise ValueError('No data key was found')
        for k in self.event_keys:
            if k in selected_data.keys():
                event_array = selected_data[k]
                break
        
        mne_data = super().generate_mne(filepath, data_array, data_type)
        # handle event and return raw
        if event_array is None:
            return mne_data
        event_array = event_array.squeeze()
        assert len(event_array.shape) == 1
        event_id = {str(i): i for i in np.unique(event_array)}
        events = np.zeros((len(event_array), 3))
        events[:, 0] = range(len(event_array))
        events[:, -1] = event_array

        raw_data = Raw(filepath, mne_data)
        raw_data.set_event(events, event_id)

        return raw_data


