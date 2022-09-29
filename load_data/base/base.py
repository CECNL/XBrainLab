import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from enum import Enum

from ...base import TopWindow, ValidateException, InitWindowValidateException
from ...dataset.data_holder import Raw
from .event import LoadEvent

class EditRaw(TopWindow): # called when double click on treeview
    def __init__(self, parent, raw):
        super().__init__(parent, "Edit data attribute")
        self.raw = raw
        self.check_data()

        # ==== init
        self.delete_row = False
        self.events = self.event_id = None
        self.subject_var = tk.StringVar(self)
        self.session_var = tk.StringVar(self)

        tk.Label(self, text="Filepath: ").grid(row=0, column=0, sticky='w')
        tk.Label(self, text="Filename: ").grid(row=1, column=0, sticky='w')
        tk.Label(self, text="Subject: ").grid(row=2, column=0, sticky='w')
        tk.Label(self, text="Session: ").grid(row=3, column=0, sticky='w')
        tk.Label(self, text="Channels: ").grid(row=4, column=0, sticky='w')
        tk.Label(self, text="Sampling Rate: ").grid(row=5, column=0, sticky='w')
        tk.Label(self, text="Epochs: ").grid(row=6, column=0, sticky='w')
        tk.Label(self, text="Events: ").grid(row=7, column=0, sticky='w')

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
        events, event_id = LoadEvent(self, self.raw).get_result()
        if event_id:
            self.events = events
            self.event_id = event_id
            self.event_label.config(text=','.join(str(e) for e in event_id))
    
    def confirm(self, *args):
        if not self.subject_var.get():
            raise ValidateException('Subject name cannot be empty')
        if not self.session_var.get():
            raise ValidateException('Subject name cannot be empty')
        if self.event_id:
            try:
                self.raw.set_event(self.events, self.event_id)
            except:
                raise ValidateException(self, f'Inconsistent number of events with epochs length (got {len(self.events)})')
        self.raw.set_subject_name(self.subject_var.get())
        self.raw.set_session_name(self.session_var.get())
        self.destroy()
    
    def _get_result(self):
        return self.delete_row

class DataType(Enum):
    RAW = 'raw'
    EPOCH = 'epochs'

class LoadBase(TopWindow):
    def __init__(self, parent, title, lock_config_status=False):
        # ==== initialize ====
        super().__init__(parent, title)
        self.ret_val = None
        self.filetypes = ()
        self.lock_config_status = lock_config_status

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
        attr_header = ["Filename", "Subject", "Session", "Channels", "Sampling Rate", "Epochs", "Events"]
        self.data_attr_treeview = ttk.Treeview(attr_frame, columns=attr_header, show='headings', selectmode="browse") # filepath not displayed
        for h in attr_header:
            self.data_attr_treeview.column(h, width=len(h)*8+10, anchor=tk.CENTER) # for setting width
            self.data_attr_treeview.heading(h, text=h, anchor=tk.CENTER)
        self.data_attr_treeview.grid(row=0, column=0)
        self.data_attr_treeview.bind('<Double-Button-1>', self.edit)

        # ==== status table ==== 
        # channels, events
        stat_frame = ttk.LabelFrame(self, text="Current Status")
        self.raw_data_len_var = tk.IntVar()
        self.event_ids_var = tk.StringVar()
        tk.Label(stat_frame, text="Dataset loaded: ").grid(row=0, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.raw_data_len_var).grid(row=0, column=1, sticky='w')
        tk.Label(stat_frame, text="Loaded type: ").grid(row=1, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.type_ctrl).grid(row=1, column=1, sticky='w')
        tk.Label(stat_frame, text="Event name: ").grid(row=2, column=0, sticky='w')
        tk.Label(stat_frame, textvariable=self.event_ids_var, wraplength=175).grid(row=2, column=1, sticky='w')

        # ==== functional buttons ====
        add_btn = tk.Button(self, text="Add", command=self.load)
        confirm_btn = tk.Button(self, text="Confirm", command=self.confirm)
        
        # ==== pack ====
        type_frame.grid(row=0, column=0, columnspan=2, sticky='w')
        stat_frame.grid(row=0, column=2, rowspan=2)
        attr_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        add_btn.grid(row=2, column=0, ipadx=3, sticky='e')
        confirm_btn.grid(row=2, column=1, ipadx=3, sticky='w')
        
        self.stat_frame = stat_frame
        self.reset()

    def reset(self):
        self.raw_data_list = []
        self.raw_data_len_var.set(0)
        self.event_ids_var.set('None')
        if not self.lock_config_status:
            self.type_raw.config(state="active")
            self.type_epoch.config(state="active")
        else:
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")
    #
    def check_loaded_data_consistency(self, raw):
        if not self.raw_data_list:
            return
        # check channel number
        if self.raw_data_list[-1].get_nchan() != raw.get_nchan():
            raise ValidateException(window=self, message=f'Dataset channel numbers inconsistent (got {raw.get_nchan()}).')
        # check sfreq
        if self.raw_data_list[-1].get_sfreq() != raw.get_sfreq():
            raise ValidateException(window=self, message=f'Dataset sample frequency inconsistent (got {raw.get_sfreq()}).')
        # check same data type
        if self.raw_data_list[-1].is_raw() != raw.is_raw():
            raise ValidateException(window=self, message=f'Dataset type inconsistent.')
        # check epoch trail size
        if not raw.is_raw():
            if self.raw_data_list[-1].get_epoch_duration() != raw.get_epoch_duration():
                raise ValidateException(window=self, message=f'Epoch duration inconsistent (got {raw.get_epoch_duration()}).')

    def get_loaded_raw(self, filepath):
        for raw_data in self.raw_data_list:
            if filepath == raw_data.get_filepath():
                return raw_data
        return None
    
    def has_record(self):
        if self.raw_data_list:
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
            if self.get_loaded_raw(filepath):
                continue
            mne_data = self._load(filepath)
            if mne_data is None:
                raise ValidateException(self, f'Unable to load {filepath}.')
            if mne_data is False:
                return
            if not isinstance(mne_data, Raw):
                raw_data = Raw(filepath, mne_data)
            else:
                raw_data = mne_data
            self.check_loaded_data_consistency(raw_data)
            self.data_attr_treeview.insert('', iid=filepath, index="end", values=raw_data.get_row_info())
            self.raw_data_list.append(raw_data)

        self.update_panel()
    #
    def check_data_type(self, data_type):
        if data_type != self.type_ctrl.get():
            if self.raw_data_list:
                raise ValidateException(self, 'Unable to load type raw and epochs at the same time')
            if data_type == DataType.RAW.value:
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 2, switch to raw loading")
            elif data_type == DataType.EPOCH.value:
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 3, switch to epochs loading")
            self.type_ctrl.set(data_type)    
    #
    def update_panel(self):
        # update dataset length
        if len(self.raw_data_list) > 0: # cannot switch once some data were loaded
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")
        elif not self.lock_config_status:
            return self.reset()
        self.raw_data_len_var.set(len(self.raw_data_list))
        # update event list
        event_list_str = []
        for raw in self.raw_data_list:
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
        raw_data = self.get_loaded_raw(selected_row)
        if not raw_data:
            return
        del_row = EditRaw(self, raw_data).get_result()
        if del_row:
            self.raw_data_list.remove(raw_data)
            self.data_attr_treeview.delete(selected_row)
        else:
            if not self.winfo_exists():
                return
            self.data_attr_treeview.item(selected_row, values=raw_data.get_row_info())
        self.update_panel()

    def confirm(self):
        # check all data with labels
        for raw_data in self.raw_data_list:
            _, event_id = raw_data.get_event_list()
            if not event_id:
                raise ValidateException(self, f"No label has been loaded for {raw_data.get_filename()}")
        self.ret_val = self.raw_data_list
        self.destroy()
    
    def _get_result(self):
        return self.ret_val