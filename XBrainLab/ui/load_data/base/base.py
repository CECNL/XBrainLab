import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from enum import Enum
import re, os

from ...base import TopWindow, ValidateException, InitWindowValidateException
from ...script import Script
from .event import LoadEvent

from XBrainLab.load_data import RawDataLoader, Raw

class DataType(Enum):
    RAW = 'raw'
    EPOCH = 'epochs'

class EditRaw(TopWindow): # called when double click on treeview
    def __init__(self, parent, raw):
        super().__init__(parent, "Edit data attribute")
        self.raw = raw
        self.check_data()
        self.columnconfigure([1], weight=1)
        self.rowconfigure(list(range(8)), weight=1)

        # ==== init
        self.delete_row = False
        self.events = self.event_id = None
        self.subject_var = tk.StringVar(self)
        self.session_var = tk.StringVar(self)

        self.script_history = Script()
        self.event_script_history = None
        self.ret_script_history = None

        tk.Label(self, text="Filepath: ").grid(row=0, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Filename: ").grid(row=1, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Subject: ").grid(row=2, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Session: ").grid(row=3, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Channels: ").grid(row=4, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Sampling Rate: ").grid(row=5, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Epochs: ").grid(row=6, column=0, sticky='w', padx=10, pady=2)
        tk.Label(self, text="Events: ").grid(row=7, column=0, sticky='w', padx=10, pady=2)

        tk.Label(self, text=raw.get_filepath()).grid(row=0, column=1, sticky='w')
        tk.Label(self, text=raw.get_filename()).grid(row=1, column=1, sticky='w')
        tk.Entry(self, textvariable=self.subject_var).grid(row=2, column=1, sticky='w')
        tk.Entry(self, textvariable=self.session_var).grid(row=3, column=1, sticky='w')
        tk.Label(self, text=raw.get_nchan()).grid(row=4, column=1, sticky='w')
        tk.Label(self, text=raw.get_sfreq()).grid(row=5, column=1, sticky='w')
        tk.Label(self, text=raw.get_epochs_length()).grid(row=6, column=1, sticky='w')
        self.event_label = tk.Label(self, text=raw.get_event_name_list_str())
        self.event_label.grid(row=7, column=1, sticky='w')
        
        tk.Button(self, text="Delete", command=self._delete_row).grid(row=8, column=0)
        tk.Button(self, text="Load Events", command=self._load_events).grid(row=8, column=1)
        tk.Button(self, text="Confirm", command=self.confirm).grid(row=8, column=2)

        self.subject_var.set(raw.get_subject_name())
        self.session_var.set(raw.get_session_name())
    
    def check_data(self):
        if not isinstance(self.raw, Raw):
            raise InitWindowValidateException(self, 'Invalid Raw data.')
    
    def _delete_row(self, *args):
        self.delete_row = True
        self.destroy()
    
    def _load_events(self, *args): # load event from file or view loaded event
        event_module = LoadEvent(self, self.raw)
        events, event_id = event_module.get_result()
        script = event_module.get_script_history()
        if event_id:
            self.event_script_history = script
            self.events = events
            self.event_id = event_id
            self.event_label.config(text=','.join(str(e) for e in event_id))
    
    def confirm(self, *args):
        if not self.subject_var.get():
            raise ValidateException(self, 'Subject name cannot be empty')
        if not self.session_var.get():
            raise ValidateException(self, 'Subject name cannot be empty')

        if self.event_id:
            try:
                self.raw.set_event(self.events, self.event_id)
                self.script_history.add_script(self.event_script_history)
                self.script_history.add_cmd(f"event_loader.apply()")
            except:
                raise ValidateException(self, f'Inconsistent number of events with epochs length (got {len(self.events)})')
        
        if self.subject_var.get() != self.raw.get_subject_name():
            self.raw.set_subject_name(self.subject_var.get())
            self.script_history.add_cmd(f"raw_data.set_subject_name({repr(self.subject_var.get())})")
        if self.session_var.get() != self.raw.get_session_name():
            self.raw.set_session_name(self.session_var.get())
            self.script_history.add_cmd(f"raw_data.set_session_name({repr(self.session_var.get())})")
        
        self.ret_script_history = self.script_history
        self.destroy()
    
    def _get_result(self):
        return self.delete_row

    def _get_script_history(self):
        return self.ret_script_history

class LoadBase(TopWindow):
    def __init__(self, parent, title, lock_config_status=False):
        # ==== initialize ====
        super().__init__(parent, title)
        self.ret_val = None
        self.ret_script_history = None
        self.filetypes = ()
        self.lock_config_status = lock_config_status

        self.columnconfigure([0], weight=2)
        self.columnconfigure([1], weight=1)
        self.rowconfigure([1], weight=1)


        # ==== type selection ==== 
        type_frame = ttk.LabelFrame(self, text="Data type")
        self.type_ctrl = tk.StringVar()
        self.type_ctrl.set(DataType.RAW.value)
        self.type_raw = tk.Radiobutton(type_frame, text="Raw", value=DataType.RAW.value, variable=self.type_ctrl)
        self.type_epoch = tk.Radiobutton(type_frame, text="Epochs", value=DataType.EPOCH.value, variable=self.type_ctrl)
        self.type_raw.grid(row=0, column=0,sticky="w")
        self.type_epoch.grid(row=0, column=1,sticky="w")

        # ==== attr table ====  (self.data_attr_treeview)
        attr_frame = ttk.LabelFrame(self, text="Data attributes")
        attr_frame.columnconfigure([0], weight=1)
        attr_frame.rowconfigure([0], weight=1)
        attr_header = ["Filename", "Subject", "Session", "Channels", "Sampling Rate", "Epochs", "Events"]
        self.data_attr_treeview = ttk.Treeview(attr_frame, columns=attr_header, show='headings', selectmode="browse") # filepath not displayed
        self.data_attr_scrollbar = tk.Scrollbar(attr_frame, orient ="vertical",command = self.data_attr_treeview.yview)
        for h in attr_header:
            self.data_attr_treeview.column(h, width=len(h)*8+10, anchor=tk.CENTER) # for setting width
            self.data_attr_treeview.heading(h, text=h, anchor=tk.CENTER)
        self.data_attr_treeview.grid(row=0, column=0, sticky='nwes')
        self.data_attr_scrollbar.grid(row=0, column=1, sticky='nse')
        self.data_attr_treeview.configure(yscrollcommand=self.data_attr_scrollbar.set)
        self.data_attr_treeview.bind('<Double-Button-1>', self.edit)

        # ==== status table ==== 
        # channels, events
        stat_frame = ttk.LabelFrame(self, text="Current Status")
        stat_frame.columnconfigure([1], weight=1)
        self.raw_data_len_var = tk.IntVar()
        self.event_ids_var = tk.StringVar()
        self.filename_template_var = tk.StringVar(self)
        tk.Label(stat_frame, text="Dataset loaded: ").grid(row=0, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.raw_data_len_var).grid(row=0, column=1, sticky='w')
        tk.Label(stat_frame, text="Loaded type: ").grid(row=1, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.type_ctrl).grid(row=1, column=1, sticky='w')
        tk.Label(stat_frame, text="Event name: ").grid(row=2, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.event_ids_var, wraplength=175).grid(row=2, column=1, sticky='w')
        tk.Label(stat_frame, text="Filename template: ").grid(row=3, column=0, sticky='w')
        tk.Entry(stat_frame, textvariable=self.filename_template_var).grid(row=4, column=0, columnspan=2, sticky='ew')
        self.stat_frame_row_count = 5

        # ==== functional buttons ====
        btn_frame = tk.Frame(self)
        add_btn = tk.Button(btn_frame, text="Add", command=self.load)
        confirm_btn = tk.Button(btn_frame, text="Confirm", command=self.confirm)
        add_btn.pack(side=tk.LEFT)
        confirm_btn.pack(side=tk.LEFT)
        
        # ==== pack ====
        type_frame.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=10)
        attr_frame.grid(row=1, column=0, sticky='news')
        stat_frame.grid(row=1, column=1, sticky='ew', padx=10)
        btn_frame.grid(row=2, column=0,  columnspan=2)
        
        self.stat_frame = stat_frame
        self.reset()

    def reset(self):
        self.data_loader = RawDataLoader()
        self.script_history = Script()
        self.script_history.add_import('import mne')
        self.script_history.add_import('from XBrainLab.load_data import Raw')
        self.script_history.add_cmd('data_loader = study.get_raw_data_loader()')
        self.script_history.newline()

        self.raw_data_len_var.set(0)
        self.event_ids_var.set('None')
        if not self.lock_config_status:
            self.type_raw.config(state="active")
            self.type_epoch.config(state="active")
        else:
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")
    #
    def has_record(self):
        if self.data_loader:
            return True
        return False

    def _load(self): # for overriding
        raise NotImplementedError

    def load(self):
        selected_files = filedialog.askopenfilenames (
            parent = self,
            filetypes = self.filetypes
        )
        for filepath in selected_files:
            if self.data_loader.get_loaded_raw(filepath):
                continue
            self.script_history.newline()
            self.script_history.add_cmd('filepath = ' + repr(filepath))
            data = self._load(filepath)
            if data is None:
                raise ValidateException(self, f'Unable to load {filepath}.')
            if data is False:
                return
            if not isinstance(data, Raw):
                self.script_history.add_cmd('raw_data = Raw(filepath, data)')
                raw_data = Raw(filepath, data)
            else:
                self.script_history.add_cmd('raw_data = data')
                raw_data = data
            try:
                self.data_loader.check_loaded_data_consistency(raw_data)
            except Exception as e:
                raise ValidateException(window=self, message=str(e))
            if self.filename_template_var.get():
                raw_data.parse_filename(regex=self.filename_template_var.get())
                self.script_history.add_cmd(f"raw_data.parse_filename(regex={repr(self.filename_template_var.get())})")
            
            self.data_attr_treeview.insert('', iid=filepath, index="end", values=raw_data.get_row_info())
            self.data_loader.append(raw_data)
            self.script_history.add_cmd("data_loader.append(raw_data)")

        self.update_panel()
    #
    def check_data_type(self, data_type):
        if data_type != self.type_ctrl.get():
            if self.data_loader:
                raise ValidateException(self, 'Unable to load type raw and epochs at the same time')
            if data_type == DataType.RAW.value:
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 2, switch to raw loading")
            elif data_type == DataType.EPOCH.value:
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 3, switch to epochs loading")
            self.type_ctrl.set(data_type)    
    #
    def update_panel(self):
        # update dataset length
        if len(self.data_loader) > 0: # cannot switch once some data were loaded
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")
        else:
            return self.reset()
        self.raw_data_len_var.set(len(self.data_loader))
        # update event list
        event_list_str = []
        for raw in self.data_loader:
            _, event_id = raw.get_event_list()
            if event_id:
                for event_name in event_id:
                    if event_name not in event_list_str:
                        event_list_str.append(str(event_name))
        if event_list_str:
            self.event_ids_var.set('\n'.join(event_list_str))
        else:
            self.event_ids_var.set('None')
    
    def edit(self, event): # open window for editing subject/session/load data on double click
        selected_row = self.data_attr_treeview.focus()
        raw_data = self.data_loader.get_loaded_raw(selected_row)
        if not raw_data:
            return
        self.script_history.newline()
        self.script_history.add_cmd(f"raw_data = data_loader.get_loaded_raw({repr(selected_row)})")
        edit_module = EditRaw(self, raw_data)
        del_row = edit_module.get_result()
        edit_script_history = edit_module.get_script_history()
        if del_row:
            self.data_loader.remove(raw_data)
            self.script_history.add_cmd("data_loader.remove(raw_data)")
            self.data_attr_treeview.delete(selected_row)
        else:
            if not self.window_exist:
                return
            self.script_history.add_script(edit_script_history)
            self.data_attr_treeview.item(selected_row, values=raw_data.get_row_info())
        self.update_panel()

    def confirm(self):
        # check all data with labels
        try:
            self.data_loader.validate()
            self.script_history.add_cmd('data_loader.validate()', newline=True)
        except ValueError as e:
            raise ValidateException(self, str(e))
        self.ret_val = self.data_loader
        self.ret_script_history = self.script_history
        self.destroy()
    
    def _get_result(self):
        return self.ret_val

    def _get_script_history(self):
        return self.ret_script_history
