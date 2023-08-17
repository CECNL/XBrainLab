import mne
import os, re
import tkinter as tk
import numpy as np
from enum import Enum
from copy import deepcopy
import traceback

from ..utils import validate_type

class FilenameGroupKey(Enum):
    SUBJECT = 'subject'
    SESSION = 'session'

class Raw:
    def __init__(self, filepath, mne_data, normalize=True):
        validate_type(filepath, str, 'filepath')
        validate_type(mne_data, (mne.io.BaseRaw, mne.BaseEpochs), 'mne_data')
        self.filepath = filepath
        if normalize:
            for ch in range(mne_data._data.shape[0]):
                mne_data._data[ch,:] = mne_data._data[ch,:]-mne_data._data[ch,:].mean() # /(mne_data._data[ch,:].max()-mne_data._data[ch,:].min())
                # mne_data._data[ch,:] = (mne_data._data[ch,:]-mne_data._data[ch,:].min())/(mne_data._data[ch,:].max()-mne_data._data[ch,:].min())
        self.mne_data = mne_data
        self.preprocess_history = []
        self.raw_events = None
        self.raw_event_id = None
        self.subject = 0
        self.session = 0
    
    def get_filepath(self):
        return self.filepath
    
    def get_filename(self):
        return os.path.basename(self.filepath)
    
    def get_subject_name(self):
        return str(self.subject)

    def get_session_name(self):
        return str(self.session)
    
    def get_preprocess_history(self):
        return self.preprocess_history
    
    def add_preprocess(self, desc):
        self.preprocess_history.append(desc)

    def parse_filename(self, regex):
        try:
            filepath = self.get_filepath()
            filename = os.path.basename(filepath)
            m = re.match(regex, filename)
            groupdict = m.groupdict()
            if FilenameGroupKey.SESSION.value in groupdict:
                self.set_session_name(groupdict[FilenameGroupKey.SESSION.value])
            if FilenameGroupKey.SUBJECT.value in groupdict:
                self.set_subject_name(groupdict[FilenameGroupKey.SUBJECT.value])
        except:
            traceback.print_exc()
            pass

    def set_subject_name(self, subject):
        self.subject = subject

    def set_session_name(self, session):
        self.session = session

    def set_event(self, events, event_id):
        validate_type(events, np.ndarray, 'events')
        validate_type(event_id, dict, 'event_id')
        assert len(events.shape) == 2 and events.shape[1] == 3
        
        if self.is_raw() and (self.raw_events is None or self.raw_events.shape == events.shape): # legal event update
            self.raw_events = events
            self.raw_event_id = event_id
        elif self.is_raw():
            raw_events_copy = self.raw_events.copy()
            events_copy = events.copy()

            uq, cnt = np.unique(self.raw_events[:,-1], return_counts=True)
            
            event_count = {u:c for u,c in zip(uq, cnt)}
            event_targets = {v:k for k,v in self.raw_event_id.items() if event_count[v]==len(events)}

            assert len(event_targets) > 0

            for target in event_targets.values():
                _ = self.raw_event_id.pop(target)
            for k,target in event_targets.items():
                for i,j in enumerate(event_id.values()):
                    new_id = 1
                    while new_id in self.raw_event_id.values():
                        new_id += 1
                    self.raw_event_id[target+f'_{i}'] = new_id
                    events[np.where(events_copy[:,-1]==j),-1] = new_id

                self.raw_events[np.where(raw_events_copy[:,-1]==k),-1] = events[:,-1]
            print(f'UserWarning: Inconsistent number of events with existing event number {self.raw_events.shape[0]} (got {len(events)}). Events with matching number are automatically updated with running numbers.')
        else:
            assert self.get_epochs_length() == len(events)
            self.mne_data.events = events
            self.mne_data.event_id = event_id

    def set_mne(self, data):
        if isinstance(data,  mne.epochs.BaseEpochs):
            if self.raw_event_id:
                assert len(self.raw_events) == len(data.events)
                data.events = self.raw_events
                data.event_id = self.raw_event_id
                self.raw_events = None
                self.raw_event_id = None
        self.mne_data = data

    def set_mne_and_wipe_events(self, data):
        self.raw_events = None
        self.raw_event_id = None
        self.mne_data = data

    #
    def get_mne(self):
        return self.mne_data

    def get_tmin(self):
        return self.mne_data.tmin

    def get_nchan(self):
        return self.mne_data.info['nchan']

    def get_sfreq(self):
        return self.mne_data.info['sfreq']

    def get_filter_range(self):
        return self.mne_data.info['highpass'], self.mne_data.info['lowpass']
        
    def get_epochs_length(self):
        if self.is_raw():
            return 1
        return len(self.mne_data.events)
    
    def get_epoch_duration(self):
        return self.mne_data.get_data().shape[-1]

    def is_raw(self):
        return isinstance(self.mne_data, mne.io.base.BaseRaw)
    #
    def get_raw_event_list(self):
        try:
            if self.raw_event_id:
                return self.raw_events, self.raw_event_id
        except:
            pass
        try:
            if self.mne_data.event_id:
                return self.mne_data.events, self.mne_data.event_id
        except:
            pass
        try:
            events = mne.find_events(self.mne_data)
            if len(events) != len(np.unique(events[:,0])):
                    events[:,0] = range(len(events))
            event_ids = {}
            for i, event in enumerate(np.unique(events[:,-1])):
                event_ids[i] = event

            self.set_event(events, event_ids)
            return self.raw_events, self.raw_event_id
        except:
            try:
                events, event_ids = mne.events_from_annotations(self.mne_data)
                self.set_event(events, event_ids)
                return self.raw_events, self.raw_event_id
            except Exception as e:
                raise RuntimeError("Exception occured in get_raw_event_list():",e)
        return None, None

    def get_event_list(self):
        if self.raw_event_id:
            return self.raw_events, self.raw_event_id
        return self.get_raw_event_list()

    def has_event(self):
        events, event_id = self.get_event_list()
        if event_id:
            return True
        return False

    def has_event_str(self):
        if self.has_event():
            return 'yes'
        return 'no'
    
    def get_event_name_list_str(self):
        if not self.has_event():
            return 'None'
        events, event_id = self.get_event_list()
        return ','.join([str(e) for e in event_id])
        
    #
    def get_row_info(self):
        channel = self.get_nchan()
        sfreq = self.get_sfreq()
        epochs = self.get_epochs_length()
        has_event = self.has_event_str()
        return self.get_filename(), self.get_subject_name(), self.get_session_name(), channel, sfreq, epochs, has_event

