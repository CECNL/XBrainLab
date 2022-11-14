import tkinter as tk
from ..base import TopWindow, InitWindowValidateException, ValidateException
import os
import numpy as np
import mne

class PickMontageWindow(TopWindow):
    command_label = 'Set Montage'
    def __init__(self, parent, channel_names):
        super().__init__(parent, self.command_label)
        self.channel_names = channel_names
        self.check_data()
        self.chs = None
        self.positions = None
        # init data
        montage_list = mne.channels.get_builtin_montages()
        
        #+ gui
        ##+ option menu
        selected_montage = tk.StringVar(self)
        selected_montage.trace('w', self.on_montage_select) # callback
        montage_opt = tk.OptionMenu(self, selected_montage, *montage_list)
        ##+ listview
        selected_label = tk.Label(self, text='Selected')
        option_list = tk.Listbox(self, selectmode=tk.EXTENDED)
        seleced_list = tk.Listbox(self, selectmode=tk.EXTENDED)

        button_frame = tk.Frame(self)
        tk.Button(button_frame, text='>>', command=self.add).pack()
        tk.Button(button_frame, text='<<', command=self.remove).pack()

        montage_opt.grid(row=0, column=0)
        selected_label.grid(row=0, column=2)
        option_list.grid(row=1, column=0, sticky='news')
        button_frame.grid(row=1, column=1)
        seleced_list.grid(row=1, column=2, sticky='news')
        tk.Button(self, text='confirm', command=self.confirm).grid(row=2, column=0, columnspan=3)
        self.columnconfigure([0,2], weight=1)
        self.rowconfigure([1], weight=1)

        
        self.selected_montage = selected_montage
        self.option_list = option_list
        self.seleced_list = seleced_list
        self.selected_label = selected_label
        self.options = None
        selected_montage.set(montage_list[0])
    
    def check_data(self):
        if not self.channel_names:
            raise InitWindowValidateException(self, 'No valid channel name is provided')

    def add(self):
        selected = self.get_selected()
        for i in self.option_list.curselection():
            if self.options[i] not in selected:
                self.seleced_list.insert(tk.END, self.options[i])
        self.selected_label.config(text=f'{self.seleced_list.size()} Selected')

    def remove(self):
        options = self.seleced_list.curselection()
        for i in options[::-1]:
            self.seleced_list.delete(i)
        self.selected_label.config(text=f'{self.seleced_list.size()} Selected')

    def get_selected(self):
        return [self.seleced_list.get(i) for i in range(self.seleced_list.size())]
            
    def on_montage_select(self, var_name, *args):
        while self.option_list.size() > 0:
            self.option_list.delete(0)
        while self.seleced_list.size() > 0:
            self.seleced_list.delete(0)
        
        montage = mne.channels.make_standard_montage(self.selected_montage.get())
        options = list(montage.get_positions()['ch_pos'].keys())
        self.options = options
        self.option_list.insert(tk.END, *options)

        for ch in self.channel_names:
            if ch in options:
                self.seleced_list.insert(tk.END, ch)
            else:
                break
        self.selected_label.config(text=f'{self.seleced_list.size()} Selected')
    
    def confirm(self):
        montage = mne.channels.make_standard_montage(self.selected_montage.get())
        chs = self.get_selected()
        if len(chs) != len(self.channel_names):
            raise ValidateException(window=self, message=f'Number of channels mismatch ({len(chs)} != {len(self.channel_names)})')
        positions = np.array([montage.get_positions()['ch_pos'][ch] for ch in chs])
        self.chs = chs
        self.positions = positions
        self.destroy()

    def _get_result(self):
        return self.chs, self.positions