import numpy as np
from enum import Enum
from ..option import SplitUnit
from copy import deepcopy

class Raw:
    def __init__(self, raw_attr={}, raw_data={}, raw_event={}, event_ids={}):
        self.raw_attr = {} # {'filename': (subject, session)}
        self.raw_events = {} # {'filename': label of (n_events, 3)}
        self.mne_data = {} # {'filename': mne struct}

        self.subjects = {} # index: subject name
        self.sessions = {} # subject name: session numbers
        
        self.sfreq = 0
        self.event_id = {}
        if raw_attr!={} and raw_data!={}:
            self.update(raw_attr=raw_attr, raw_data=raw_data, raw_event=raw_event, event_ids=event_ids)
    
    def update(self, raw_attr, raw_data, raw_event, event_ids):
        self.event_id = event_ids
        self.raw_attr = raw_attr
        self.mne_data = raw_data
        for fn in raw_attr.keys():
            self.sfreq = raw_data[fn].info['sfreq']
            self.raw_events[fn] = raw_event[fn][0] if type(raw_event[fn]) == tuple else raw_event[fn]
            if raw_attr[fn][0] not in self.subjects.values():
                self.subjects[len(self.subjects)] = raw_attr[fn][0]
                self.sessions[raw_attr[fn][0]] = 1
            else:
                self.sessions[raw_attr[fn][0]] += 1
            

    def copy(self):
        copy_event = {}
        i = 0
        for fn in self.raw_attr.keys():
            copy_event[fn] = (self.raw_events[fn].copy(), self.event_id)
            i+=1
        return Raw(self.raw_attr, deepcopy(self.mne_data), copy_event, self.event_id)