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
class RequireInput(TopWindow):
    # todo: type check ?

    """
    title: string window title
    msg: string description of expected inputs
    request: list of of string on asked inputs
    
    return: {request: input}
    """
    def __init__(self, parent, title, msg, requests):
        super(RequireInput, self).__init__(parent)
        self.title(title)
        self.msg = tk.Label(self, text=msg).grid(row=0,column=0)
        self.var = {r: tk.StringVar() for r in requests}
        self.ret = {r:None for r in requests}
        
        for r,i in zip(requests, range(len(requests))):
            tk.Label(self, text=r).grid(row=i+1, column=0)
            tk.Entry(self, textvariable=self.var[r]+": ").grid(row=i+1, column=1)
        tk.Button(self, text="Confirm", command=lambda:self.confirm(), width=10).grid(row=len(requests)+2, column=0)
    def confirm(self):
        for v in self.var.keys():
            res = self.var[v].get()
            self.ret[v] = res
        self.destroy()
    
    def _get_result(self):
        return self.ret
# data array: session, channel, timpstamp
# todo: get attr funcs for mne structures?

class Raw:
    def __init__(self, raw_attr, raw_data):
        self.id_map = {}

        self.subject = []
        self.session = []
        self.label = []
        self.data = []

        self._init_attr(raw_attr=raw_attr, raw_data=raw_data)
    
    def _init_attr(self, raw_attr, raw_data):
        i = 0
        for fn in raw_attr.keys():
            self.id_map[fn] = i
            self.subject.append(raw_attr[fn][0])
            self.session.append(raw_attr[fn][1])
            self.data.append(raw_data[fn])

    def inspect(self):
        for k,v in self.id_map.items():
            print(k, self.subject[v], self.session[v])
            print(self.data[v])
    
class Epochs:
    def __init__(self, epoch_attr, epoch_data):
        self.id_map = {}

        self.subject = []
        self.session = []
        self.label = []
        self.data = []

        self._init_attr(epoch_attr=epoch_attr, epoch_data=epoch_data)
    
    def _init_attr(self, epoch_attr, epoch_data):
        i = 0
        for fn in epoch_attr.keys():
            self.id_map[fn] = i
            self.subject.append(epoch_attr[fn][0])
            self.session.append(epoch_attr[fn][1])
            self.data.append(epoch_data[fn])

    def inspect(self):
        for k,v in self.id_map.items():
            print(k, self.subject[v], self.session[v])
            print(self.data[v])



        