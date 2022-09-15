import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from typing_extensions import IntVar
from collections import OrderedDict
import numpy as np
import scipy.io
import mne
import os

class TopWindow(tk.Toplevel):
    def __init__(self, parent, title):
        #self._window = tk.Toplevel()
        tk.Toplevel.__init__(self)
        self.parent = parent
        self.title(title)
    def _get_result(self):
        """Override this to return values."""
        return None
        
    def get_result(self):
        try:
            self.wait_window()
        except:
            pass
        
        return self._get_result()

# todo: get attr funcs for mne structures?

class Raw:
    """
    raw_attr: {fn: [subject, session]}
    raw_data: {fn: mne.io.Raw}
    raw_event: {fn: [labels, event_ids]}
    """
    def __init__(self, raw_attr, raw_data, raw_event):
        self.id_map = {} # {fn: subject/session/data list position id}
        self.event_id_map = {} # {fn: label list position id}

        self.subject = []
        self.session = []
        self.label = []
        self.data = []

        self.event_id = {}
        self._init_attr(raw_attr=raw_attr, raw_data=raw_data, raw_event=raw_event)
    
    def _init_attr(self, raw_attr, raw_data, raw_event):
        i = 0
        for fn in raw_attr.keys():
            self.id_map[fn] = i
            self.subject.append(raw_attr[fn][0])
            self.session.append(raw_attr[fn][1])
            self.data.append(raw_data[fn])
            if fn in raw_event.keys():
                self.event_id_map[fn] = i
                self.label.append(raw_event[fn][0])
                if self.event_id == {}:
                    self.event_id = raw_event[fn][1]
                else:
                    assert self.event_id == raw_event[fn][1], 'Event id inconsistent.'
            i += 1

            
    def inspect(self):
        for k,v in self.id_map.items():
            #print(k, self.subject[v], self.session[v])
            print(self.data[v])
            #print(len(self.label[v]))
        print(self.event_id)
        #print(self.label)
        #print(self.event_id_map)
    
class Epochs:
    """
    epoch_attr: {fn: [subject, session]}
    epoch_data: {fn: mne.Epochs}
    """
    def __init__(self, epoch_attr, epoch_data):
        self.id_map = {}

        self.subject = []
        self.session = []
        self.label = []
        self.data = []

        self.event_id = {}

        self._init_attr(epoch_attr=epoch_attr, epoch_data=epoch_data)
    
    def _init_attr(self, epoch_attr, epoch_data):
        i = 0
        for fn in epoch_attr.keys():
            self.id_map[fn] = i
            self.subject.append(epoch_attr[fn][0])
            self.session.append(epoch_attr[fn][1])
            self.data.append(epoch_data[fn])
            
            self.label.append(epoch_data[fn].events[:,2]) 
            if self.event_id=={}:
                self.event_id = epoch_data[fn].event_id
            else:
                assert epoch_data[fn].event_id == self.event_id, 'Event Id inconsistent.'

    def inspect(self):
        for k,v in self.id_map.items():
            #print(k, self.subject[v], self.session[v])
            print(self.data[v])
            print(self.event_id)


        