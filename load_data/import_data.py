from ..base.top_window import TopWindow
from ..dataset.data_holder import Raw, Epochs
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from typing_extensions import IntVar
from collections import OrderedDict
import numpy as np
import scipy.io
import mne


class _loadevent(TopWindow):
    # TODO:
    #      event file saved format & encoding & file type(lacking test data)
    """
    parameter: parent, title
               fn: filename
    return: boolean of events were successfully loaded
    """

    def __init__(self, parent, title, fn):
        # ==== init
        super(_loadevent, self).__init__(parent, title)
        self.loaded_event = () # get returned events from mne functions
        self.label_list = []
        self.ret_event_loaded = False

        # ==== view event loaded
        event_src = 0
        event_src_msg = ['Load event file.', 'Event parsed from stimulus channel.', 'Event parsed from annotation.', 'View events']
        self.event_num = tk.StringVar()
        self.event_num.set('None')
        self.event_id_dict = tk.StringVar()
        self.event_id_dict.set('None')

        # ==== try parse from data (stimulus channel & annotation)
        if self.parent.parent.type_ctrl.get() == 'epochs':
            self.loaded_event = (self.parent.parent.data_list[fn].events, self.parent.parent.data_list[fn].event_id)
            event_src = 3
        else:
            try:
                self.loaded_event = mne.find_events(self.parent.parent.data_list[fn])
                event_src = 1
            except:
                try:
                    self.loaded_event = mne.events_from_annotations(self.parent.parent.data_list[fn])
                    if self.loaded_event[1]!={}: # edf could generate empty events
                        event_src = 2
                except (ValueError,TypeError) as err:
                    pass
        if self.loaded_event[1]=={'1':1} and self.parent.parent.type_ctrl.get() == 'epochs':
            event_src = 0
        # ==== view
        srcframe = tk.LabelFrame(self, text=event_src_msg[event_src])
        srcframe.grid(row=0, column=0)
        if event_src == 0:
            tk.Button(srcframe, text="Load file", command=lambda:self._load_event_file(fn)).grid(row=0, column=0)
        else:
            tk.Button(srcframe, text="Load file", command=lambda:self._load_event_file(fn), state=tk.DISABLED).grid(row=0, column=0)
            self.ret_event_loaded = True
            self.event_num.set(str(self.loaded_event[0].shape[0]))
            self.event_id_dict.set(str(self.loaded_event[1]))
            if self.parent.parent.type_ctrl.get() == 'raw':
                self.parent.parent.raw_events[fn] = [self.loaded_event[0][:,2], self.loaded_event[1]]
        
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

class _editrow(TopWindow):
    """
    parameter: parent, title
               header: attr value fields (table header + 'Filepath')
               target: attr dict (see init_row)
    return: boolean of whether current row is to be deleted
    """
    def __init__(self, parent, title, header, target):
        super().__init__(parent, title)
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
    
    def _load_events(self, fn, target):
        self.event_loaded = _loadevent(self,"Load events", fn).get_result()
        if self.event_loaded == True:
            target['Events'].set('yes')
        
    def _delete_row(self):
        self.delete_row = True
        self.destroy()
    
    def _confirm_val(self, target):
        if self.event_loaded == True:
            if self.parent.type_ctrl.get() == 'raw' and self.parent.event_ids.get() == 'None':
                self.parent.event_ids.set(str(self.parent.raw_events[target['Filename'].get()][1]))
            elif self.parent.event_ids.get()=='None':
                self.parent.event_ids.set(str(self.parent.data[target['Filename'].get()][1]))
        for h in target.keys():
            target[h].set(target[h].get())
        self.result = target.copy()
        self.destroy()
    
    def _get_result(self):
        return self.delete_row

class LoadTemplate(TopWindow):
    # TODO: 
    #       hover cursor shows full filepath msg (optional)
    """
    parameter: parent
    return: Raw or Epochs class object defined in Template.py
    """
    def __init__(self, parent, title):
        # ==== initialize ==== 
        super().__init__(parent,title)
        self.attr_list = OrderedDict() # fn: attr_row
        self.data_list = OrderedDict() # fn: mne struct

        self.ret_val = None # Raw or Epoch
        self.raw_events = {} # if events of raw structure were loaded, fn: [label of (n_event,1), event_ids dict] 
        self.attr_row_template = self._init_row()

        self.attr_info = {'data key': [], 'event key': [], 'sampling rate': 0, 'nchan': 0, 'ntimes':0} # used for .mat & .npy/npz

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
        self.data_attr_treeview = ttk.Treeview(attr_frame, columns=self.attr_header[1:], show='headings') # filepath not displayed
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
        self.event_ids = tk.StringVar()
        self.event_ids.set('None')
        tk.Label(self.stat_frame, text="Dataset loaded: ").grid(row=0, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.attr_len).grid(row=0, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Loaded type: ").grid(row=1, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.type_ctrl).grid(row=1, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Event id: ").grid(row=2, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.event_ids).grid(row=2, column=1, sticky='w')

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

    def _make_row(self, selected_path, selected_data, raw_event_bool=0):
        new_row = self._init_row()
        new_row['Filepath'].set(selected_path)
        new_row['Filename'].set(selected_path.split('/')[-1])
        new_row['Channels'].set(int(selected_data.info['nchan']))
        new_row['Sampling Rate'].set(int(selected_data.info['sfreq']))
        if self.type_ctrl.get() == 'raw':
            if(len(selected_data.info['events'])) == 0:
                new_row['Epochs'].set(1) # raw: only 1 epoch
            if not raw_event_bool:
                new_row['Events'].set('yes')
        else:
            new_row['Epochs'].set(len(selected_data.events))
            new_row['Events'].set('yes')
        return new_row
    
    def _edit_row(self, event):
        selected_row = self.data_attr_treeview.focus()
        selected_row = self.data_attr_treeview.item(selected_row)
        del_row = _editrow(self, "Edit data attribute", self.attr_header, self.attr_list[selected_row['text']]).get_result()
        self.data_attr_treeview.delete(*self.data_attr_treeview.get_children())
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
            self.attr_info = {'data key': [], 'event key': [], 'sampling rate': 0, 'nchan': 0, 'ntimes':0}
            self.nch.set(self.attr_info['nchan'])
            self.srate.set(self.attr_info['sampling rate'])
            self.type_raw.config(state="active")
            self.type_epoch.config(state="active")
        
        for k,v in self.attr_list.items():
            v_val = tuple([vv.get() for vv in v.values()])
            self.data_attr_treeview.insert('', index="end" ,text=k, values=v_val[1:])
    
    def _load(self):
        # for overriding
        pass
    def _list_update(self, attr_list_tmp, data_list_tmp, raw_event_tmp={}):
        for k,v in attr_list_tmp.items():
            if k not in self.attr_list.keys():
                v_val = tuple([vv.get() for vv in v.values()])
                self.data_attr_treeview.insert('', index="end" ,text=k, values=v_val[1:])
                self.attr_list[k] = v
                self.data_list[k] = data_list_tmp[k]
            # events
            if self.type_ctrl.get()=='epochs':
                if self.event_ids.get() == 'None':
                    self.event_ids.set(str(data_list_tmp[k].event_id))
                else:
                    assert str(data_list_tmp[k].event_id)==self.event_ids.get(), 'Event ids inconsistent.'
            elif k in raw_event_tmp.keys():
                if self.event_ids.get() == 'None':
                    self.event_ids.set(str(raw_event_tmp[k][1]))
                else:
                    assert str(raw_event_tmp[k][1])==self.event_ids.get(), 'Event ids inconsistent.'
        self.raw_events.update(raw_event_tmp)
        self.attr_len.set(len(self.attr_list))

        if len(self.attr_list) != 0:
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")

    def _data_from_array(self, data_array, data_info, attr_info_tmp, selected_data):
        if self.type_ctrl.get() == 'raw':
            assert len(data_array.shape) == 2, 'Data dimension invalid.'
            selected_data = mne.io.RawArray(data_array, data_info)
            if attr_info_tmp['event key'] != []:
                assert sum(k in attr_info_tmp['event key'] for k in selected_data.keys())<=1, 'Data has multiple keys identified as containing event.'
                for k in attr_info_tmp['event key']:
                    if k in selected_data.keys():
                        event_label = selected_data[k]
                event_label = list(set(event_label.flatten()))
                event_id = {str(i): event_label[i] for i in range(len(event_label))}
                return selected_data, (event_label, event_id)
        else:
            assert len(data_array.shape) == 3, 'Data dimension invalid.'
            selected_data_tmp = mne.EpochsArray(data_array, data_info)
            if attr_info_tmp['event key'] != []:
                assert sum(k in attr_info_tmp['event key'] for k in selected_data.keys())<=1, 'Data has multiple keys identified as containing event.'
                for k in attr_info_tmp['event key']:
                    if k in selected_data.keys():
                        event_label = selected_data[k]
                selected_data_tmp.events[:,2] = np.squeeze(event_label)
                event_label = list(set(event_label.flatten()))
                selected_data_tmp.event_id = {str(i): event_label[i] for i in range(len(event_label))}
            selected_data = selected_data_tmp
        return selected_data, None
            

    def _reshape_array(self, target, attr_info_tmp): # used for .mat & .npy/npz
        # shape for feeding raw constructor: (channel, timepoint)
        # shape for feeding epoch constructor: (epoch, channel, timepoint)
        target = np.squeeze(target) # squeeze dimension of 1, should ended up be 3d
        target_shape = [s for s in target.shape]
        dim_ch = target_shape.index(attr_info_tmp['nchan'])
        dim_time = target_shape.index(attr_info_tmp['ntimes'])
        if len(target_shape) == 3:
            return np.transpose(target, (3-dim_ch-dim_time, dim_ch, dim_time))
        else:
            return np.transpose(target, (dim_ch, dim_time))
    def _clear_key(self):
        self.attr_info['data key'] = []
        self.attr_info['event key'] = []
    
    def _confirm(self):
        # check channel number
        check_chs = set(self.data_attr_treeview.item(at_row)['values'][3] for at_row in self.data_attr_treeview.get_children())
        assert len(check_chs)==1, 'Dataset channel numbers inconsistent.'

        ret_attr = {}
        ret_data = {}
        for at_row in self.data_attr_treeview.get_children():
            at_val = self.data_attr_treeview.item(at_row)['values']
            ret_attr[at_val[0]] = [at_val[1], at_val[2]] # subject, session
            ret_data[at_val[0]] = self.data_list[at_val[0]] # mne
            
        if self.type_ctrl.get() == 'raw':
            self.ret_val = Raw(ret_attr, ret_data, self.raw_events)
        else:
            self.ret_val = Epochs(ret_attr, ret_data)
        self.destroy()
    
    def _get_result(self):
        return self.ret_val

class LoadSet(LoadTemplate):
    command_label = "Import SET file (EEGLAB toolbox)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .set files")
   
    def _load(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.set'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys():
                if self.type_ctrl.get() == 'raw':
                    selected_data = mne.io.read_raw_eeglab(fn, uint16_codec='latin1', preload=True)
                else:
                    selected_data = mne.io.read_epochs_eeglab(fn, uint16_codec='latin1')
                new_row = self._make_row(fn, selected_data)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        # update attr table
        self._list_update(attr_list_tmp, data_list_tmp)

class _loadmat(TopWindow):
    # TODO: spinbox default (if cleaner method is possible)
    """
    parameter: parent, title
               fp: filepath
               attr_info_tmp: current attr dict of load window
               loaded_mat: mat file content (in dict)
    return: attr dict with selected key (from spinbox) and nchan/srate/ntime (input or already exist)
    """
    def __init__(self, parent, title, fp, attr_info_tmp, loaded_mat):
        # ==== inits
        super().__init__(parent, title)
        
        self.loaded_mat = loaded_mat
        self.ret_key = attr_info_tmp
        self.srate = tk.IntVar()
        self.srate.set(attr_info_tmp['sampling rate'])
        self.nch = tk.IntVar()
        self.nch.set(attr_info_tmp['nchan'])
        self.ntime = tk.IntVar()
        self.ntime.set(attr_info_tmp['ntimes'])
        tk.Label(self, text="Filename: "+fp.split('/')[-1]).grid(row=0, column=0, sticky='w')

        # ==== select key
        spin_val = [k for k in loaded_mat.keys() if k not in ['__globals__', '__version__', '__header__']]
        spin_val = tuple(reversed(spin_val)) 
        
        self.data_shape_view = tk.StringVar()
        self.data_key_trace = tk.StringVar()
        self.event_shape_view = tk.StringVar()
        self.event_key_trace = tk.StringVar()

        # ======== data key
        data_key_frame = ttk.LabelFrame(self, text="Select Data key")
        data_key_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        
        tk.Label(data_key_frame, text="Data key: ").grid(row=0, column=0, sticky='w')
        self.data_key_select = tk.Spinbox(data_key_frame, values = spin_val, textvariable=self.data_key_trace)
        self.data_key_select.grid(row=0, column=1, sticky='w')

        self.data_shape_view.set(str(self.loaded_mat[self.data_key_select.get()].shape)) # init spinbox 1
        
        tk.Label(data_key_frame, text="Value shape: ").grid(row=1, column=0, sticky='w')
        tk.Label(data_key_frame, textvariable=self.data_shape_view).grid(row=1, column=1)
        self.data_key_trace.trace_variable('r', self._shape_view_update)

        # ======== horizontal line
        sep = ttk.Separator(self, orient='horizontal')
        sep.grid(row=2)

        # ======== event key
        event_key_frame = ttk.LabelFrame(self, text="Select Event key")
        event_key_frame.grid(row=3, column=0, columnspan=2, sticky='w')
        
        tk.Label(event_key_frame, text="Event key: ").grid(row=0, column=0, sticky='w')
        self.event_key_select = tk.Spinbox(event_key_frame, values = ('None',)+spin_val, textvariable=self.event_key_trace) # need the comma for expressing as tuple
        self.event_key_select.grid(row=0, column=1, sticky='w')
        
        self.event_key_trace.set('None') # init spinbox 2
        self.event_shape_view.set('None') # init view shape 2
        
        tk.Label(event_key_frame, text="Value shape: ").grid(row=1, column=0, sticky='w')
        tk.Label(event_key_frame, textvariable=self.event_shape_view).grid(row=1, column=1)
        self.event_key_trace.trace_variable('r', self._shape_view_update)

        # ==== sampling rate & channel
        tk.Label(self, text="Sampling Rate: ").grid(row=4, column=0, sticky='w')
        tk.Label(self, text="Channel: ").grid(row=5, column=0, sticky='w')
        tk.Label(self, text="Time samples: ").grid(row=6, column=0, sticky='w')

        sr_ch_time = [self.srate, self.nch, self.ntime]
        if attr_info_tmp['nchan'] == 0:
            for tv in sr_ch_time:
                tk.Entry(self, textvariable=tv).grid(row=4+sr_ch_time.index(tv), column=1, sticky='w')
        else:
            for t in sr_ch_time:
                tk.Label(self, text=t.get()).grid(row=4+sr_ch_time.index(t), column=1, sticky='w')

        # ==== confirm
        tk.Button(self, text="Confirm",command=self._key_confirm).grid(row=7, column=0)

    def _shape_view_update(self, var, id, mode): # on spinbox change
        self.data_shape_view.set(str(self.loaded_mat[self.data_key_select.get()].shape)) 
        if self.event_key_select.get() == 'None':
            self.event_shape_view.set('None')
        else:
            self.event_shape_view.set(str(self.loaded_mat[self.event_key_select.get()].shape))
        
    def _key_confirm(self):
        self.ret_key["data key"].append(self.data_key_select.get())
        self.ret_key["event key"].append(self.event_key_select.get())
        self.ret_key["sampling rate"] = self.srate.get()
        self.ret_key["nchan"] = self.nch.get()
        self.ret_key["ntimes"] = self.ntime.get()

        assert self.ret_key["data key"][-1] != self.ret_key["event key"][-1], 'Data key and event key should be different.'
        assert self.ret_key["sampling rate"] >0, 'Sampling rate invalid.'
        assert (self.ret_key["nchan"] >0) and (self.ret_key["nchan"] in self.loaded_mat[self.data_key_select.get()].shape), 'Number of channels invalid.'
        assert (self.ret_key["ntimes"] >0) and (self.ret_key["ntimes"] in self.loaded_mat[self.data_key_select.get()].shape), 'Number of time points invalid.'
        self.destroy()

    def _get_result(self):
        return self.ret_key

class _keyhandler(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, "Selected keys")

        

class LoadMat(LoadTemplate):
    # TODO: reset common settings
    command_label = "Import MAT file (Matlab array)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .mat files")
        self.data_attr_treeview.bind('<Double-Button-1>', self._edit_row_attr)

        # ==== status table ====
        self.nch = tk.IntVar()
        self.nch.set(0)
        self.srate = tk.IntVar()
        self.srate.set(0)
        tk.Label(self.stat_frame, text="Channels: ").grid(row=3, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.nch).grid(row=3, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Sampling rate: ").grid(row=4, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.srate).grid(row=4, column=1, sticky='w')
        tk.Button(self.stat_frame, text="Clear selected keys", command=lambda:self._clear_key()).grid(row=5, column=0)

    def _load(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.mat'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}
        raw_event_tmp = {}
        attr_info_tmp = self.attr_info

        for fn in selected_tuple:            
            if fn.split('/')[-1] not in self.attr_list.keys():
                selected_data = scipy.io.loadmat(fn)

                if (self.attr_info['nchan'] ==0 and attr_info_tmp['nchan']==0 ) \
                    or any(k in selected_data.keys() for k in attr_info_tmp['data key'])==False \
                    or any(k in selected_data.keys() for k in attr_info_tmp['event key'])==False:
                    attr_info_tmp = _loadmat(self, "Select Field", fn,attr_info_tmp, selected_data).get_result()

                assert sum(k in attr_info_tmp['data key'] for k in selected_data.keys())<=1, 'Data has multiple keys identified as containing data.'
                
                for k in attr_info_tmp['data key']:
                    if k in selected_data.keys():
                        data_array = selected_data[k]
                        break
                data_array = self._reshape_array(data_array, attr_info_tmp)
                data_info = mne.create_info(attr_info_tmp['nchan'], attr_info_tmp['sampling rate'], 'eeg')
                
                selected_data, rawevent_tmp = self._data_from_array(data_array, data_info, attr_info_tmp, selected_data)
                if rawevent_tmp != {}:
                    raw_event_tmp[fn] = rawevent_tmp
                      
                new_row = self._make_row(fn, selected_data, rawevent_tmp==None)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data

        # update attr table
        self._list_update(attr_list_tmp, data_list_tmp, raw_event_tmp)
        if self.nch.get()==0:
            self.attr_info = attr_info_tmp
            self.nch.set(self.attr_info['nchan'])
            self.srate.set(self.attr_info['sampling rate'])

class LoadEdf(LoadTemplate):
    command_label = "Import EDF/EDF+ file (BIOSIG toolbox)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .edf files")
        self.type_raw.config(state="disabled")# only supporting raw
        self.type_epoch.config(state="disabled")

    def _load(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.edf'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys(): 
                selected_data = mne.io.read_raw_edf(fn, preload=True)
                new_row = self._make_row(fn, selected_data)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        self._list_update(attr_list_tmp, data_list_tmp)

class LoadCnt(LoadTemplate):
    # TODO: untested, lack of test data
    # combine with load edf???
    command_label = "Import CNT file (Neuroscan)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .cnt files")
        self.type_raw.config(state="disabled")# only supporting raw
        self.type_epoch.config(state="disabled")


    def _load(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.cnt'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys(): 
                selected_data = mne.io.read_raw_cnt(fn, preload=True)
                new_row = self._make_row(fn, selected_data)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        self._list_update(attr_list_tmp, data_list_tmp)

class _loadnpy(TopWindow):
    """
    parameter: parent, title
               fp: filepath
               attr_info: current attr dict of load window
               loaded_array: np file content (in np array)
    return: attr dict with selected key (None) and nchan/srate/ntime (input)
    """
    def __init__(self, parent, title, fp, attr_info, loaded_array):
        super().__init__(parent, title)
        self.loaded_array = loaded_array
        self.ret_key = attr_info
        
        self.attr_dict = { k : tk.IntVar() for k in ["Channel", "Sampling rate", "Time samples"]}
        tk.Label(self, text="Filename: "+fp.split('/')[-1]).grid(row=0, column=0, sticky='w')
        tk.Label(self, text="Current shape: "+str(loaded_array.shape)).grid(row=1, column=0, sticky='w')
        i = 2
        for k in self.attr_dict.keys():
            tk.Label(self, text = k+": ").grid(row=i, column=0, sticky='w')
            tk.Entry(self, textvariable=self.attr_dict[k]).grid(row=i,column=1, sticky='w')
            i += 1
        tk.Button(self,text="Confirm", command=lambda:self._confirm()).grid(row=i, column=0)

    def _confirm(self):
        self.ret_key['sampling rate'] = self.attr_dict['Sampling rate'].get()
        self.ret_key['nchan'] = self.attr_dict['Channel'].get()
        self.ret_key['ntimes'] = self.attr_dict['Time samples'].get()
        assert self.ret_key['nchan'] in self.loaded_array.shape, 'Channel number invalid.'
        assert self.ret_key['ntimes'] in self.loaded_array.shape, 'Time sample number number invalid.'
        assert self.ret_key['sampling rate'] >0, 'Sampling rate invalid.'
        self.destroy()
    
    def _get_result(self):
        return self.ret_key

class LoadNp(LoadTemplate):
    # npy: single array
    # npz: multiple arrays
    command_label = "Import NPY/NPZ file (Numpy array)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .npy/.npz files")
        self.data_attr_treeview.bind('<Double-Button-1>', self._edit_row_attr)

        # ==== status table ====
        self.nch = tk.IntVar()
        self.nch.set(0)
        self.srate = tk.IntVar()
        self.srate.set(0)
        tk.Label(self.stat_frame, text="Channels: ").grid(row=3, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.nch).grid(row=3, column=1, sticky='w')
        tk.Label(self.stat_frame, text="Sampling rate: ").grid(row=4, column=0, sticky='w')
        tk.Label(self.stat_frame, textvariable=self.srate).grid(row=4, column=1, sticky='w')
        tk.Button(self.stat_frame, text="Clear selected keys", command=lambda:self._clear_key()).grid(row=5, column=0)
    
    def _load(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.npy'), ('eeg file', '*.npz'))
        )
        attr_list_tmp = {}
        data_list_tmp = {}
        raw_event_tmp = {}
        attr_info_tmp = self.attr_info

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys():
                selected_data = np.load(fn)
                if isinstance(selected_data, np.lib.npyio.NpzFile): # npz
                    selected_data = {k:selected_data[k] for k in selected_data.keys()}
                    if (self.attr_info['nchan'] ==0 and attr_info_tmp['nchan']==0 ) \
                    or any(k in selected_data.keys() for k in attr_info_tmp['data key'])==False \
                    or any(k in selected_data.keys() for k in attr_info_tmp['event key'])==False:
                        attr_info_tmp.update(_loadmat(self, "Set attribute", fn, attr_info_tmp, selected_data).get_result())
                else: # npy
                    if self.attr_info['nchan'] ==0 and attr_info_tmp['nchan']==0:
                        attr_info_tmp.update(_loadnpy(self, "Set attribute", fn, attr_info_tmp, selected_data).get_result())
                
                if isinstance(selected_data, dict): # npz
                    assert sum(k in attr_info_tmp['data key'] for k in selected_data.keys())<=1, 'Data has multiple keys identified as containing data.'
                    for k in attr_info_tmp['data key']:
                        if k in selected_data.keys():
                            data_array = selected_data[k]
                            break
                else:
                    data_array = selected_data
                
                data_array = self._reshape_array(data_array, attr_info_tmp)
                data_info = mne.create_info(attr_info_tmp['nchan'], attr_info_tmp['sampling rate'], 'eeg')
                
                selected_data, rawevent_tmp = self._data_from_array(data_array, data_info, attr_info_tmp, selected_data)
                if rawevent_tmp != {}:
                    raw_event_tmp[fn] = rawevent_tmp
                      
                new_row = self._make_row(fn, selected_data, rawevent_tmp==None)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        self._list_update(attr_list_tmp, data_list_tmp, raw_event_tmp)
        if self.nch.get()==0:
            self.attr_info = attr_info_tmp
            self.nch.set(self.attr_info['nchan'])
            self.srate.set(self.attr_info['sampling rate'])
        
