import numpy as np
import mne

from ...script import Script

from .base import DataType
from XBrainLab.load_data import Raw

class RawInfo:
    def __init__(self):
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

    def reshape_array(self, data, script):
        reshape_idx = []
        if len(data.shape) > 2:
            epoch_idx = self.channel_info.index(self.channel_type.EPOCH)
            reshape_idx = [epoch_idx]
        ch_idx = self.channel_info.index(self.channel_type.CH)
        time_idx = self.channel_info.index(self.channel_type.TIME)
        reshape_idx += [ch_idx, time_idx]
        data = np.transpose(data, reshape_idx)
        script.add_cmd(f"data = np.transpose(data, axes={repr(reshape_idx)})")
        return data

    def generate_mne(self, filepath, data_array, data_type):
        script = Script()
        script.add_import("import numpy as np")
        script.add_import("import mne")
        data_array = self.reshape_array(data_array, script)
        data_info = mne.create_info(self.nchan, self.sfreq, 'eeg')
        script.add_cmd((
            "data_info = mne.create_info("
            f"{repr(self.nchan)}, {repr(self.sfreq)}, 'eeg')"
        ))
        
        if data_type == DataType.RAW.value:
            mne_data = mne.io.RawArray(data_array, data_info)
            script.add_cmd("data = mne.io.RawArray(data, data_info)")
        elif data_type == DataType.EPOCH.value:
            mne_data = mne.EpochsArray(
                data=data_array, info=data_info, tmin=self.tmin
            )
            script.add_cmd((
                "data = mne.EpochsArray("
                f"data=data, info=data_info, tmin={repr(self.tmin)})"
            ))
        return mne_data, script
   
class DictInfo(RawInfo):
    def __init__(self):
        super().__init__()
        self.data_keys = set()
        self.event_keys = set()

    def copy(self):
        new_dict_info = DictInfo()
        new_dict_info.data_keys = self.data_keys.copy()
        new_dict_info.event_keys = self.event_keys.copy()
        return new_dict_info

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
        script = Script()
        script.add_import("import numpy as np")
        script.add_import("import mne")
        data_array = event_array = None

        for k in self.event_keys:
            if k in selected_data.keys():
                event_array = selected_data[k]
                script.add_cmd(f"event = data[{repr(k)}]")
                break

        for k in self.data_keys:
            if k in selected_data.keys():
                data_array = selected_data[k]
                script.add_cmd(f"data = data[{repr(k)}]")
                break
                
        if data_array is None:
            raise ValueError('No data key was found')
        
        
        mne_data, array_script = super().generate_mne(
            filepath, data_array, data_type
        )
        script += array_script
        # handle event and return raw
        if event_array is None:
            return mne_data, script
        event_array = event_array.squeeze()
        assert len(event_array.shape) == 1
        event_id = {str(i): i for i in np.unique(event_array)}
        events = np.zeros((len(event_array), 3))
        events[:, 0] = range(len(event_array))
        events[:, -1] = event_array

        raw_data = Raw(filepath, mne_data)
        raw_data.set_event(events, event_id)

        script.add_cmd("event = event.squeeze()")
        script.add_cmd("event_id = {str(i): i for i in np.unique(event)}")
        script.add_cmd("events = np.zeros((len(event), 3))")
        script.add_cmd("events[:, 0] = range(len(event))")
        script.add_cmd("events[:, -1] = event")
        script.add_cmd("data = Raw(filepath, data)")
        script.add_cmd("data.set_event(events, event_id)")

        return raw_data, script


