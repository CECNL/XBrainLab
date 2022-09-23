from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from collections import OrderedDict
import numpy as np
import scipy.io
import mne
from copy import deepcopy

def parse_event(current_data): # for raw
    src_event = 0
    ret_event = () # (3d event, event_id)
    try:
        ret_event = mne.find_events(current_data)
        src_event = 1
    except:
        try:
            ret_event = mne.events_from_annotations(current_data)
            src_event = 2
            if ret_event[1] == {}:
                ret_event = ()
                src_event = 0
        except: 
            pass
    return ret_event, src_event

class _loadevent(TopWindow): # called from edit_row
    # TODO: event file saved format & encoding & file type(lacking test data)
    def __init__(self, parent, title, fn):
        # ==== init
        super(_loadevent, self).__init__(parent, title)
        self.loaded_event = () # get returned events from mne functions
        self.label_list = []
        self.ret_event_loaded = False

        # ==== for viewing event loaded
        event_src = 0
        event_src_msg = ['Load event file.', 'Event parsed from stimulus channel.', 'Event parsed from annotation.', 'View events']
        self.event_num = tk.StringVar()
        self.event_num.set('None')
        self.event_id_dict = tk.StringVar()
        self.event_id_dict.set('None')

        # ==== get events from epochs data or get events already parsed from raw data
        if self.parent.parent.type_ctrl.get() == 'epochs':
            self.loaded_event = (self.parent.parent.data_list[fn].events, self.parent.parent.data_list[fn].event_id)
            event_src = 3
            if self.loaded_event[1] == {'1':1}:
                self.loaded_event = ()
                event_src = 0
        elif fn in self.parent.parent.raw_event_src.keys():
            self.loaded_event = self.parent.parent.raw_events[fn]
            event_src = self.parent.parent.raw_event_src[fn]

        # ==== view event loaded
        srcframe = tk.LabelFrame(self, text=event_src_msg[event_src])
        srcframe.grid(row=0, column=0)
        if event_src == 0:
            tk.Button(srcframe, text="Load file", command=lambda:self._load_event_file(fn)).grid(row=0, column=0)
        else:
            tk.Button(srcframe, text="Load file", command=lambda:self._load_event_file(fn), state=tk.DISABLED).grid(row=0, column=0)
            self.ret_event_loaded = True
            self.event_num.set(str(self.loaded_event[0].shape[0]))
            self.event_id_dict.set(str(self.loaded_event[1]))
        
        tk.Label(srcframe, text="Event numbers: ").grid(row=1, column=0, sticky='w')
        tk.Label(srcframe, textvariable=self.event_num).grid(row=1, column=1, sticky='w')
        tk.Label(srcframe, text="Event id: ").grid(row=2, column=0, sticky='w')
        tk.Label(srcframe, textvariable=self.event_id_dict).grid(row=2, column=1, sticky='w')
        
        tk.Button(self, text="Confirm", command=lambda:self._confirm(fn)).grid(row=1, column=0)
    
    def _load_event_file(self, fn):
        selected_file = filedialog.askopenfilename(
            parent = self,
            filetypes = (
                ('text file', '*.txt'),
                #('mat file', '*mat'),
                #('text file', '*.lst'),
                #('text file', '*.eve'),
                #('binary file', '*.fif')
            )
        )
        with open(selected_file, encoding='utf-8', mode='r') as fp:
            for line in fp.readlines():
                self.label_list += [int(l.rstrip()) for l in line.split(' ')] # for both (n,1) and (1,n) of labels
            fp.close()
        
    def _confirm(self,fn):
        event_id_dict = {str(i): list(set(self.label_list))[i] for i in range(len(list(set(self.label_list))))} # make event id dict with label named with index
        if self.loaded_event == ():
            if self.parent.parent.type_ctrl.get() == 'raw': # mne raw has no event & event_id attributes
                self.parent.parent.raw_events[fn] = [self.label_list, event_id_dict]
            else: # set mne epochs event data
                self.parent.parent.data_list[fn].events[:,2] = self.label_list
                self.parent.parent.data_list[fn].event_id = event_id_dict

        # set load event window display
        self.event_num.set(str(len(self.label_list)))
        self.event_id_dict.set(str(event_id_dict))
        self.destroy()

    def _get_result(self):
        return self.ret_event_loaded

class _editrow(TopWindow): # called when double click on treeview
    def __init__(self, parent, title, header, target):
        super().__init__(parent, title)
        self._check_data(target)

        # ==== init
        self.delete_row = False
        self.event_loaded = True
        
        i = 0
        for h in header:
            if h in ["Filepath", "Filename", "Channels", "Sampling Rate", "Epochs", "Events"]:
                tk.Label(self, text=h+": ").grid(row=i, column=0, sticky='w')
                tk.Label(self, textvariable=target[h]).grid(row=i, column=1, sticky='w')
            else:
                tk.Label(self, text=h+": ").grid(row=i, column=0, sticky='w')
                tk.Entry(self, text=target[h]).grid(row=i, column=1, sticky='w')
            i += 1
        tk.Button(self, text="Delete", command=lambda:self._delete_row()).grid(row=i, column=0)
        tk.Button(self, text="Load Events", command=lambda:self._load_events(target['Filename'].get(), target)).grid(row=i, column=1)
        tk.Button(self, text="Confirm", command=lambda:self._confirm_val(target)).grid(row=i, column=2)
    
    def _check_data(self, target):
        if not isinstance(target, dict):
            raise InitWindowValidateException(self, '_edit_row target parameter should be dict of a row in treeview.')
    
    def _load_events(self, fn, target): # load event from file or view loaded event
        self.event_loaded = _loadevent(self,"Load events", fn).get_result()
        if self.event_loaded == True:
            target['Events'].set('yes')
        
    def _delete_row(self):
        self.delete_row = True
        self.destroy()
    
    def _confirm_val(self, target):
        if self.event_loaded == True:
            if self.parent.type_ctrl.get() == 'raw':
                if self.parent.event_ids_var.get() == 'None':
                    self.parent.event_ids = self.parent.raw_events[target['Filename'].get()][1]
                else:
                    self.parent.event_ids.update(self.parent.raw_events[target['Filename'].get()][1])
            else:
                if self.parent.event_ids_var.get()=='None':
                    self.parent.event_ids = self.parent.data[target['Filename'].get()][1]
                else:
                    self.parent.event_ids.update(self.parent.data[target['Filename'].get()][1])
            self.parent.event_ids_var.set(str(self.parent.event_ids))
        for h in target.keys():
            target[h].set(target[h].get())
        self.result = target.copy()
        self.destroy()
    
    def _get_result(self):
        return self.delete_row

class _loadmat(TopWindow):
    def __init__(self, parent, title, fp, attr_info_tmp, loaded_mat):
        # ==== inits
        super().__init__(parent, title)
        self._check_data(fp, attr_info_tmp, loaded_mat)
        
        self.loaded_mat = loaded_mat
        self.ret_key = attr_info_tmp
        self.attr_var = {k:tk.IntVar() for k in attr_info_tmp.keys() if not k in ['data key', 'event key']}
        self.attr_var['tmin'] = tk.StringVar()
        [self.attr_var[k].set(v) for k,v in attr_info_tmp.items() if not k in ['data key', 'event key']]

        tk.Label(self, text="Filename: "+fp.split('/')[-1]).grid(row=0, column=0, sticky='w')

        # ==== select key
        opt_val = [k for k in loaded_mat.keys() if k not in ['__globals__', '__version__', '__header__']]
        opt_val = tuple(opt_val) 
        
        self.data_shape_view = tk.StringVar()
        self.data_key_trace = tk.StringVar()
        self.data_key_trace.set(opt_val[0])
        self.event_shape_view = tk.StringVar()
        self.event_key_trace = tk.StringVar()
        self.event_key_trace.set(opt_val[0])

        # ======== data key
        data_key_frame = ttk.LabelFrame(self, text="Select Data key")
        data_key_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        
        tk.Label(data_key_frame, text="Data key: ").grid(row=0, column=0, sticky='w')
        self.data_key_select = tk.OptionMenu(data_key_frame, self.data_key_trace, *opt_val)
        self.data_key_select.grid(row=0, column=1, sticky='w')

        self.data_shape_view.set(str(self.loaded_mat[self.data_key_trace.get()].shape)) # init spinbox 1
        
        tk.Label(data_key_frame, text="Value shape: ").grid(row=1, column=0, sticky='w')
        tk.Label(data_key_frame, textvariable=self.data_shape_view).grid(row=1, column=1)
        self.data_key_trace.trace_add('write', self._shape_view_update)

        # ======== horizontal line
        sep = ttk.Separator(self, orient='horizontal')
        sep.grid(row=2)

        # ======== event key
        event_key_frame = ttk.LabelFrame(self, text="Select Event key")
        event_key_frame.grid(row=3, column=0, columnspan=2, sticky='w')
        
        tk.Label(event_key_frame, text="Event key: ").grid(row=0, column=0, sticky='w')
        self.event_key_select = tk.OptionMenu(event_key_frame, self.event_key_trace, 'None', *opt_val) # need the comma for expressing as tuple
        self.event_key_select.grid(row=0, column=1, sticky='w')
        
        self.event_key_trace.set('None') # init optmenu for event
        self.event_shape_view.set('None') # init shape view for event
        
        tk.Label(event_key_frame, text="Value shape: ").grid(row=1, column=0, sticky='w')
        tk.Label(event_key_frame, textvariable=self.event_shape_view).grid(row=1, column=1)
        self.event_key_trace.trace_add('write', self._shape_view_update)

        # ==== sampling rate & channel
        i = 4
        row_label = ["Sampling Rate: ", "Channel: ", "Time samples: ", "Time before event"]
        for v in self.attr_var.values():
            tk.Label(self, text = row_label[i-4]).grid(row=i, column=0, sticky='w')
            if v.get() == 0 or v.get() == '0':
                tk.Entry(self, textvariable=v).grid(row=i, column=1, sticky='w')
            else:
                tk.Label(self, text=v.get()).grid(row=i, column=1, sticky='w')
            i+=1
            if self.parent.type_ctrl.get() == 'raw' and i==7: # raw has no epoch start settings
                v.set = '0'
                break

        # ==== confirm
        tk.Button(self, text="Confirm",command=self._key_confirm).grid(row=i, column=0)

    def _check_data(self, fp, attr_info_tmp, loaded_mat):
        check_bools = [type(fp)==str, type(attr_info_tmp)==dict, type(loaded_mat)==dict]
        if not all(check_bools):
            raise InitWindowValidateException(self, 'Invalid data type passed to _loadmat on index{}'.format(\
                ','.join([str(idx) for idx in np.where(check_bools)])))

    def _shape_view_update(self, var, id, mode): # on spinbox change
        self.data_shape_view.set(str(self.loaded_mat[self.data_key_trace.get()].shape)) 
        if self.event_key_trace.get() == 'None':
            self.event_shape_view.set('None')
        else:
            self.event_shape_view.set(str(self.loaded_mat[self.event_key_trace.get()].shape))
        
    def _key_confirm(self):
        check_bools = np.array([self.data_key_trace.get() != self.event_key_trace.get(), self.attr_var['sfreq'].get() >0, \
            (self.attr_var['nchan'].get() >0) and (self.attr_var['nchan'].get() in self.loaded_mat[self.data_key_trace.get()].shape),\
            (self.attr_var['ntimes'].get() >0) and (self.attr_var['ntimes'].get() in self.loaded_mat[self.data_key_trace.get()].shape),\
            True,\
            float(self.attr_var['tmin'].get()) or self.attr_var['tmin'].get()=='0'])
        if self.event_key_trace.get()!= 'None':
            check_bools[4]=(1 in self.loaded_mat[self.event_key_trace.get()].shape)
        check_msg = np.array(['Data key and event key should be different.', 'Sampling rate invalid.',\
            'Number of channels invalid.', 'Number of time points invalid.',\
            'Event dimension invalid.', 'Tmin value invalid'])
        if not all(check_bools):
            raise ValidateException(window=self, message=' '.join(check_msg[check_bools!=True]))
        self.ret_key["data key"].append(self.data_key_trace.get())

        if self.event_key_trace.get() != 'None':
            self.ret_key["event key"].append(self.event_key_trace.get())

        keys = ["sfreq", "nchan", "ntimes", "tmin"]
        for k in keys:
            self.ret_key[k] = self.attr_var[k].get()

        self.destroy()

    def _get_result(self):
        return self.ret_key

class _key_handler(TopWindow):
    def __init__(self, parent, attr_info):
        super().__init__(parent, "Selected keys")
        self.ret_attr = deepcopy(attr_info)
        self.attr = deepcopy(attr_info)

        self.dk_bools = {dk:tk.BooleanVar() for dk in self.attr['data key']}
        self.ek_bools = {ek:tk.BooleanVar() for ek in self.attr['event key']}

        self.data_key_frame = tk.LabelFrame(self, text="Data keys")
        self.data_key_frame.grid(row=0, column=0, columnspan=3, sticky='w')
        self.event_key_frame = tk.LabelFrame(self, text="Event keys")
        self.event_key_frame.grid(row=1, column=0, columnspan=3, sticky='w')
        self._front_update()

        tk.Button(self, text="Delete", command=lambda:self._delete_key()).grid(row=2, column=0, sticky='w')
        tk.Button(self, text="Reset", command=lambda:self._reset_key()).grid(row=2, column=1, sticky='w')
        tk.Button(self, text="Confirm", command=lambda:self._confirm_key()).grid(row=2, column=2, sticky='w')
    
    def _frame_forget(self):
        for stuff in self.data_key_frame.winfo_children():
            stuff.destroy()
        for stuff in self.event_key_frame.winfo_children():
            stuff.destroy()
    
    def _front_update(self):
        self._frame_forget()
        i = 0
        if len(self.attr['data key'])==0:
            tk.Label(self.data_key_frame, text="No key selected for data.").grid(row=0, column=0, sticky='w')
        for dk in self.attr['data key']:
            if dk=='None':
                continue
            tk.Checkbutton(self.data_key_frame, text=dk, variable=self.dk_bools[dk]).grid(row=i, column=0, sticky='w')
            i+=1

        if len(self.attr['event key'])==0:
            tk.Label(self.event_key_frame, text="No key selected for event.").grid(row=0, column=0, sticky='w')
        i = 0
        for ek in self.attr['event key']:
            if ek =='None':
                continue
            tk.Checkbutton(self.event_key_frame, text=ek, variable=self.ek_bools[ek]).grid(row=i, column=0, sticky='w')
            i+=1

    def _get_result(self):
        return self.ret_attr

    def _delete_key(self):
        for k,v in self.dk_bools.copy().items():
            if v.get():
                self.attr['data key'].pop(self.attr['data key'].index(k))
                self.dk_bools.pop(k)
        for k,v in self.ek_bools.copy().items():
            if v.get():
                self.attr['event key'].pop(self.attr['event key'].index(k))
                self.ek_bools.pop(k)
        self._front_update()

    def _confirm_key(self):
        self.ret_attr = deepcopy(self.attr)
        self.destroy()

    def _reset_key(self):
        self.attr = deepcopy(self.ret_attr)
        self.dk_bools = {dk:tk.BooleanVar() for dk in self.attr['data key']}
        self.ek_bools = {ek:tk.BooleanVar() for ek in self.attr['event key']}
        self._front_update()

class _loadnpy(TopWindow):
    def __init__(self, parent, title, fp, attr_info_tmp, loaded_array):
        super().__init__(parent, title)
        self._check_data(fp, attr_info_tmp, loaded_array)

        self.loaded_array = np.squeeze(loaded_array) #avoid redundent dimension of 1
        self.ret_key = attr_info_tmp
        row_label = ["Sampling Rate: ", "Channel: ", "Time samples: ", "Time before event"]
        
        self.attr_var = {k:tk.IntVar() for k in attr_info_tmp.keys() if not k in ['data key', 'event key']}
        self.attr_var['tmin'] = tk.StringVar()
        [self.attr_var[k].set(v) for k,v in attr_info_tmp.items() if not k in ['data key', 'event key']]

        tk.Label(self, text="Filename: "+fp.split('/')[-1]).grid(row=0, column=0, sticky='w')
        tk.Label(self, text="Current shape: "+str(loaded_array.shape)).grid(row=1, column=0, sticky='w')
        
        if len(self.loaded_array.shape) == 2:
            self.attr_var['nchan'].set(self.loaded_array.shape[0])
            self.attr_var['ntimes'].set(self.loaded_array.shape[1])
            self.attr_var['tmin'].set('0')
        
        i = 2
        for v in self.attr_var.values():
            tk.Label(self, text = row_label[i-2]).grid(row=i, column=0, sticky='w')
            if v.get() == 0 or v.get() == '0':
                tk.Entry(self, textvariable=v).grid(row=i, column=1, sticky='w')
            else:
                tk.Label(self, text=v.get()).grid(row=i, column=1, sticky='w')
            i+=1
            if self.parent.type_ctrl.get() == 'raw' and i==5:
                break

        tk.Button(self,text="Confirm", command=lambda:self._confirm()).grid(row=i, column=0)
    
    def _check_data(self, fp, attr_info_tmp, loaded_array):
        check_bools = [type(fp)==str, type(attr_info_tmp)==dict, type(loaded_array).__module__ == np.__name__]
        if not all(check_bools):
            raise InitWindowValidateException(self, 'Invalid data type passed to _loadnpy on index{}'.format(\
                ','.join([str(idx) for idx in np.where(check_bools)])))

    def _confirm(self):
        check_bools = np.array([self.attr_var['nchan'].get() in self.loaded_array.shape,\
            self.attr_var['ntimes'].get() in self.loaded_array.shape,\
            self.attr_var['sfreq'].get() >0])
        check_msg = np.array(['Channel number invalid.', 'Time sample number number invalid.','Sampling rate invalid.'])
        if not all(check_bools):
            raise ValidateException(window=self, message=' '.join(check_msg[check_bools!=True]))
        self.ret_key['sfreq'] = self.attr_var['sfreq'].get()
        self.ret_key['nchan'] = self.attr_var['nchan'].get()
        self.ret_key['ntimes'] = self.attr_var['ntimes'].get()
        self.ret_key['tmin'] = self.attr_var['tmin'].get()
        self.destroy()
    
    def _get_result(self):
        return self.ret_key

class LoadTemplate(TopWindow):
    def __init__(self, parent, title):
        # ==== initialize ====
        super().__init__(parent,title)
        self.attr_list = OrderedDict() # fn: attr_row
        self.data_list = OrderedDict() # fn: mne struct

        self.ret_val = None # Raw or Epoch

        self.raw_event_src = {} # fn : src (see parse_event def)
        self.raw_events = {} # if events of raw structure were loaded, fn: [label of (n_event,1), event_ids dict] 

        self.attr_row_template = self._init_row()
        self.attr_info = {'data key': [], 'event key': [], 'sfreq': 0, 'nchan': 0, 'ntimes':0, 'tmin':'0'} # used for .mat & .npy/npz

        # ==== type selection ==== 
        type_frame = ttk.LabelFrame(self, text="Data type")
        type_frame.grid(row=0, column=0, columnspan=2, sticky='w')

        self.type_ctrl = tk.StringVar()
        self.type_raw = tk.Radiobutton(type_frame, text="Raw", value='raw', variable=self.type_ctrl)
        self.type_raw.grid(row=0, column=0,sticky="w")
        self.type_epoch = tk.Radiobutton(type_frame, text="Epochs", value='epochs', variable=self.type_ctrl)
        self.type_epoch.grid(row=0, column=1,sticky="w")
        self.type_ctrl.set('raw')

        # ==== attr table ====  (self.data_attr_treeview)
        attr_frame = ttk.LabelFrame(self, text="Data attributes")
        attr_frame.grid(row=1, column=0, columnspan=2, sticky='w')

        self.attr_header = [k for k in self.attr_row_template.keys()]
        self.data_attr_treeview = ttk.Treeview(attr_frame, columns=self.attr_header[1:], show='headings', selectmode="browse") # filepath not displayed
        [self.data_attr_treeview.column(h, width=len(h)*8+10, anchor=tk.CENTER) for h in self.attr_header[1:]] # for setting width
        [self.data_attr_treeview.heading(h, text=h, anchor=tk.CENTER) for h in self.attr_header[1:]] 
        self.data_attr_treeview.grid(row=0, column=0)
        self.data_attr_treeview.bind('<Double-Button-1>', self._edit_row)

        # ==== status table ==== 
        # channels, events
        self.stat_frame = ttk.LabelFrame(self, text="Current Status")
        self.stat_frame.grid(row=0, column=2, rowspan=2)
        self.attr_len = tk.IntVar()
        self.attr_len.set(0)
        self.event_ids = {}
        self.event_ids_var = tk.StringVar()
        self.event_ids_var.set('None')
        tk.Label(self.stat_frame, text="Dataset loaded: ").grid(row=0, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.attr_len).grid(row=0, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Loaded type: ").grid(row=1, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.type_ctrl).grid(row=1, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Event id: ").grid(row=2, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.event_ids_var, wraplength=175).grid(row=2, column=1, sticky='w')

        # ==== functional buttons ====
        self.add_btn = tk.Button(self, text="Add", command=lambda:self._load()).grid(row=2, column=0, ipadx=3, sticky='e')
        self.confirm_btn = tk.Button(self, text="Confirm", command=lambda:self._confirm()).grid(row=2, column=1, ipadx=3, sticky='w')

    def _init_row(self):
        new_row = OrderedDict()
        new_row.update({k: tk.StringVar() for k in ["Filepath", "Filename"]})
        new_row.update({k: tk.IntVar() for k in ["Subject", "Session", "Channels", "Sampling Rate", "Epochs"]})
        new_row['Events'] = tk.StringVar()
        new_row['Events'].set('no')
        return new_row

    def _make_row(self, selected_path, selected_data): # make treeview row
        new_row = self._init_row()
        new_row['Filepath'].set(selected_path)
        new_row['Filename'].set(selected_path.split('/')[-1])
        new_row['Channels'].set(int(selected_data.info['nchan']))
        new_row['Sampling Rate'].set(int(selected_data.info['sfreq']))
        if self.type_ctrl.get() == 'raw':
            raw_event_tuple, event_src = parse_event(selected_data) # try parse event from raw data
            if(len(selected_data.info['events'])) == 0:
                new_row['Epochs'].set(1) # raw: only 1 epoch
            if raw_event_tuple != () and event_src>0:
                new_row['Events'].set('yes')
            return new_row, raw_event_tuple, event_src
        else:
            new_row['Epochs'].set(len(selected_data.events))
            if selected_data.event_id != {'1':1}:
                new_row['Events'].set('yes')
            return new_row, None, -1
    
    def _edit_row(self, event): # open window for editing subject/session/load data on double click
        selected_row = self.data_attr_treeview.focus()
        selected_row = self.data_attr_treeview.item(selected_row)
        del_row = _editrow(self, "Edit data attribute", self.attr_header, self.attr_list[selected_row['text']]).get_result()
        self.data_attr_treeview.delete(*self.data_attr_treeview.get_children()) # more convenient then maintaining order
        if del_row:
            self.attr_list.pop(selected_row['text'])
            self.data_list.pop(selected_row['text'])
            self.attr_len.set(len(self.attr_list))
        if len(self.attr_list) == 0:
            self.type_raw.config(state="active")
            self.type_epoch.config(state="active")
        
        for k,v in self.attr_list.items():
            v_val = tuple([vv.get() for vv in v.values()])
            self.data_attr_treeview.insert('', index="end" ,text=k, values=v_val[1:])
    
    def _edit_row_attr(self, event): # used for .mat & .npy/npz
        selected_row = self.data_attr_treeview.focus()
        selected_row = self.data_attr_treeview.item(selected_row)
        del_row = _editrow(self, "Edit data attribute", self.attr_header, self.attr_list[selected_row['text']]).get_result()
        self.data_attr_treeview.delete(*self.data_attr_treeview.get_children())
        if del_row:
            self.attr_list.pop(selected_row['text'])
            self.data_list.pop(selected_row['text'])
            self.attr_len.set(len(self.attr_list))
        if len(self.attr_list) == 0:
            self.attr_info = {'data key': [], 'event key': [], 'sfreq': 0, 'nchan': 0, 'ntimes':0}

            self.type_raw.config(state="active")
            self.type_epoch.config(state="active")
        
        for k,v in self.attr_list.items():
            v_val = tuple([vv.get() for vv in v.values()])
            self.data_attr_treeview.insert('', index="end" ,text=k, values=v_val[1:])
    
    def _load(self): # for overriding
        pass
    
    def _list_update(self, attr_list_tmp, data_list_tmp, raw_event_tmp={}, raw_event_src_tmp={}):
        for k,v in attr_list_tmp.items():
            # events
            if self.type_ctrl.get()=='epochs': # epochs
                if self.event_ids_var.get() == 'None' and data_list_tmp[k].event_id !={'1':1}:
                    self.event_ids = data_list_tmp[k].event_id
                elif data_list_tmp[k].event_id !={'1':1}:
                    self.event_ids.update(data_list_tmp[k].event_id)
                if self.event_ids!={}:
                    self.event_ids_var.set(str(self.event_ids))
            
            elif k in raw_event_tmp.keys(): # raw with event
                if self.event_ids_var.get() == 'None':
                    self.event_ids = raw_event_tmp[k][1]
                else:
                    self.event_ids.update(raw_event_tmp[k][1])
                self.event_ids_var.set(str(self.event_ids))

            # add new treeview row
            if k not in self.attr_list.keys():
                v_val = tuple([vv.get() for vv in v.values()])
                self.data_attr_treeview.insert('', index="end" ,text=k, values=v_val[1:])
                self.attr_list[k] = v
                self.data_list[k] = data_list_tmp[k]
 
        if raw_event_src_tmp!={} and raw_event_tmp!={}:
            self.raw_events.update(raw_event_tmp)
            self.raw_event_src.update(raw_event_src_tmp)

        self.attr_len.set(len(self.attr_list))

        if len(self.attr_list) != 0: # cannot switch once some data were loaded
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")

    def _data_from_array(self, data_array, data_info, attr_info_tmp, selected_data): # for .mat & .np
        event_label = []
        if self.type_ctrl.get() == 'raw':
            selected_data_tmp = mne.io.RawArray(data_array, data_info)
            if attr_info_tmp['event key'] != []:
                if sum(k in attr_info_tmp['event key'] for k in selected_data.keys())>1:
                    raise ValidateException(window=self, message='Data has multiple keys identified as containing events.')
                for k in attr_info_tmp['event key']:
                    if k in selected_data.keys():
                        event_label = selected_data[k]
                        break
                event_label = list(set(event_label.flatten()))
                event_id = {str(i): event_label[i] for i in range(len(event_label))}
                return selected_data_tmp, (event_label, event_id)
            selected_data = selected_data_tmp
        else:
            selected_data_tmp = mne.EpochsArray(data = data_array, info=data_info, tmin=float(attr_info_tmp['tmin']))
            if attr_info_tmp['event key'] != []:
                if sum(k in attr_info_tmp['event key'] for k in selected_data.keys())>1:
                    raise ValidateException(window=self, message='Data has multiple keys identified as containing events.')         
                for k in attr_info_tmp['event key']:
                    if k in selected_data.keys():
                        event_label = selected_data[k]
                        selected_data_tmp.events[:,2] = np.squeeze(event_label)
                        event_label = list(set(event_label.flatten()))
                        selected_data_tmp.event_id = {str(i): event_label[i] for i in range(len(event_label))}
                        return selected_data_tmp, (event_label, selected_data_tmp.event_id)
            selected_data = selected_data_tmp
        return selected_data, None
            

    def _reshape_array(self, target, attr_info_tmp): # for .mat & .npy/npz
        # shape for feeding raw constructor: (channel, timepoint)
        # shape for feeding epoch constructor: (epoch, channel, timepoint)
        target = np.squeeze(target) # squeeze dimension of 1, should ended up be 3d
        target_shape = [s for s in target.shape]
        if len(target_shape) == 2:
            if self.type_ctrl.get()=='epochs': # wrong type, auto switch
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 2, switch to raw loading")
                self.type_ctrl.set('raw')
                return self._reshape_array(target, attr_info_tmp)
            return target          
        else:
            if self.type_ctrl.get() == 'raw': # wrong type, auto switch
                tk.messagebox.showwarning(parent=self, title="Warning", message="Detected data of dimension 3, switch to epochs loading")
                self.type_ctrl.set('epochs')
                return self._reshape_array(target, attr_info_tmp)
            dim_ch = target_shape.index(attr_info_tmp['nchan'])
            dim_time = target_shape.index(attr_info_tmp['ntimes'])
            return np.transpose(target, (3-dim_ch-dim_time, dim_ch, dim_time))
            
    def _clear_key(self):
        w = _key_handler(self,self.attr_info).get_result()
        if w != self.attr_info:
            self.attr_info = w
    
    def _confirm(self):
        # TODO: prevent adding data with different channel number
        # check channel number
        check_chs = set(self.data_attr_treeview.item(at_row)['values'][3] for at_row in self.data_attr_treeview.get_children())
        if len(check_chs)!=1:
            raise ValidateException(window=self, message='Dataset channel numbers inconsistent.')

        # make event_ids start from 0
        if min(self.event_ids.items()) != 0 or (max(self.event_ids.items()) + 1 != len(self.event_ids)):
            i = 0
            event_id_map = {}
            for name, idx in self.event_ids.items():
                self.event_ids[name] = i
                event_id_map[idx] = i
                i += 1
            if self.type_ctrl.get() == 'epochs':
                for k, v in self.data_list.items():
                    self.data_list[k].event_id = self.event_ids
                    for old_e, new_e in event_id_map.items():
                        self.data_list[k].events[:,2][self.data_list[k].events[:,2]==old_e] = new_e
            else:
                for fn in self.data_list.keys():
                    new_events = self.raw_events[fn][0]
                    for old_e, new_e in event_id_map.items():
                        new_events[:,2][new_events[:,2]==old_e] = new_e
                    self.raw_events[fn] = (new_events, self.event_ids)

        ret_attr = {}
        ret_data = {}
        for at_row in self.data_attr_treeview.get_children():
            at_val = self.data_attr_treeview.item(at_row)['values']
            ret_attr[at_val[0]] = [at_val[1], at_val[2]] # subject, session
            ret_data[at_val[0]] = self.data_list[at_val[0]] # mne
        if self.type_ctrl.get() == 'raw':
            self.ret_val = Raw(ret_attr, ret_data, self.raw_events, self.event_ids)
        else:
            self.ret_val = Epochs(ret_attr, ret_data)
        self.destroy()
    
    def _get_result(self):
        return self.ret_val