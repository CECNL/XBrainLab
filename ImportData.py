from tkinter import CENTER
from turtle import heading, width
from Template import *

# TODO: consistency of multiple files (partially done)
#       event
#       layout

class _editrow(TopWindow): # for editing data attributes
    def __init__(self, parent, title, header, target):
        super(_editrow, self).__init__(parent, title)
        self.delete_row = False
        
        i = 0
        for h in header:
            if h in ["Filepath", "Filename", "Channels", "Sampling Rate", "Epochs"]:
                tk.Label(self, text=h+": ").grid(row=i, column=0)
                tk.Label(self, text=target[h].get()).grid(row=i, column=1)
            else:
                tk.Label(self, text=h+": ").grid(row=i, column=0)
                tk.Entry(self, text=target[h]).grid(row=i, column=1)
            i += 1
        tk.Button(self, text="Delete", command=lambda:self._delete_row()).grid(row=i, column=0)
        tk.Button(self, text="Load Events", ).grid(row=i, column=1)
        tk.Button(self, text="Confirm", command=lambda:self._confirm_val(target)).grid(row=i, column=2)
    
    def _load_events(self):
        pass

    def _delete_row(self):
        self.delete_row = True
        self.destroy()
    
    def _confirm_val(self, target):
        for h in target.keys():
            target[h].set(target[h].get())
        self.result = target.copy()
        self.destroy()
    
    def _get_result(self):
        return self.delete_row

class LoadTemplate(TopWindow):
    def __init__(self, parent, title):
        # TODO: hover cursor shows full filepath msg

        # ==== initialize ==== 
        super(LoadTemplate, self).__init__(parent, title)
        self.attr_list = OrderedDict() # fn: attr_row
        self.data_list = OrderedDict() # fn: mne struct
        self.ret_val = None # Raw or Epoch
        self.attr_row_template = self._init_row()

        # ==== type selection ==== 
        type_frame = ttk.LabelFrame(self, text="Data type")
        type_frame.grid(row=0, column=0, columnspan=2, sticky='w')
        self.type_ctrl = tk.StringVar()
        self.type_raw = tk.Radiobutton(type_frame, text="Raw", value='raw', variable=self.type_ctrl)
        self.type_raw.grid(row=0, column=0,sticky="w")
        self.type_epoch = tk.Radiobutton(type_frame, text="Epochs", value='epochs', variable=self.type_ctrl)
        self.type_epoch.grid(row=0, column=1,sticky="w")
        self.type_ctrl.set('raw')

        # ==== attr table ====  (self.data_attr)
        attr_frame = ttk.LabelFrame(self, text="Data attributes")
        attr_frame.grid(row=1, column=0, columnspan=2, sticky='w')
        self.attr_header = [k for k in self.attr_row_template.keys()]
        self.data_attr = ttk.Treeview(attr_frame, columns=self.attr_header[1:], show='headings') # filepath not displayed
        [self.data_attr.column(h, width=len(h)*8+10, anchor=CENTER) for h in self.attr_header[1:]] # for setting width
        [self.data_attr.heading(h, text=h, anchor=CENTER) for h in self.attr_header[1:]] 
        self.data_attr.grid(row=0, column=0)
        self.data_attr.bind('<Double-Button-1>', self._edit_row)

        # ==== status table ==== 
        # channels, events
        stat_frame = ttk.LabelFrame(self, text="Current Status")
        stat_frame.grid(row=0, column=2, rowspan=2)
        self.attr_len = tk.IntVar()
        tk.Label(stat_frame, text="Dataset loaded: ").grid(row=0, column=0)
        tk.Label(stat_frame, textvariable=self.attr_len).grid(row=0, column=1)
        tk.Label(stat_frame, text="Loaded type: ").grid(row=1, column=0)
        tk.Label(stat_frame, textvariable=self.type_ctrl).grid(row=1, column=1)

        # -------- subclass differences --------
        #tk.Button(self, text="Add", command=lambda:self._load_set()).grid(row=2, column=0, ipadx=3, sticky='w')
        #tk.Button(self, text="Confirm", command=lambda:self._confirm_set()).grid(row=2, column=1, ipadx=3, sticky='w')

    def _init_row(self):
        new_row = OrderedDict()
        new_row.update({k: tk.StringVar() for k in ["Filepath", "Filename"]})
        new_row.update({k: tk.IntVar() for k in ["Subject", "Session", "Channels", "Sampling Rate", "Epochs"]})
        new_row['Events'] = tk.StringVar()
        new_row['Events'].set('no')
        return new_row

    def _make_row(self, selected_path, selected_data):
        new_row = self._init_row()
        new_row['Filepath'].set(selected_path)
        new_row['Filename'].set(selected_path.split('/')[-1])
        new_row['Channels'].set(int(selected_data.info['nchan']))
        new_row['Sampling Rate'].set(int(selected_data.info['sfreq']))
        new_row['Epochs'].set(len(selected_data.info['events']))
        if(len(selected_data.info['events'])) == 0:
            new_row['Epochs'].set(1) # raw: only 1 epoch     
        return new_row
    
    def _edit_row(self, event):
        selected_row = self.data_attr.focus()
        selected_row = self.data_attr.item(selected_row)
        del_row = _editrow(self, "Edit data attribute", self.attr_header, self.attr_list[selected_row['text']]).get_result()
        self.data_attr.delete(*self.data_attr.get_children())
        if del_row:
            self.attr_list.pop(selected_row['text'])
            self.data_list.pop(selected_row['text'])
            self.attr_len.set(len(self.attr_list))
        
        for k,v in self.attr_list.items():
            v_val = tuple([vv.get() for vv in v.values()])
            self.data_attr.insert('', index="end" ,text=k, values=v_val[1:])
    def _get_result(self):
        return self.ret_val

class LoadSet(LoadTemplate):
    def __init__(self, parent, title):
        super(LoadSet, self).__init__(parent, title)
        tk.Button(self, text="Add", command=lambda:self._load_set()).grid(row=2, column=0, ipadx=3, sticky='w')
        tk.Button(self, text="Confirm", command=lambda:self._confirm_set()).grid(row=2, column=1, ipadx=3, sticky='w')
    
    def _load_set(self):
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
        for k,v in attr_list_tmp.items():
            if k not in self.attr_list.keys():
                v_val = tuple([vv.get() for vv in v.values()])
                self.data_attr.insert('', index="end" ,text=k, values=v_val[1:])
                self.attr_list[k] = v
                self.data_list[k] = data_list_tmp[k]
        self.attr_len.set(len(self.attr_list))

        if len(self.attr_list) != 0:
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")

    def _confirm_set(self):
        # check channel number
        check_chs = set(self.data_attr.item(at_row)['values'][3] for at_row in self.data_attr.get_children())
        assert len(check_chs)==1, 'Dataset channel numbers inconsistent.'
        
        ret_attr = {}
        ret_data = {}
        for at_row in self.data_attr.get_children():
            at_val = self.data_attr.item(at_row)['values']
            ret_attr[at_val[0]] = [at_val[1], at_val[1]]
            ret_data[at_val[0]] = self.data_list[at_val[0]]
        if self.type_ctrl.get() == 'raw':
            self.ret_val = Raw(ret_attr, ret_data)
        else:
            self.ret_val = Epochs(ret_attr, ret_data)
        self.destroy()

class _loadmat(TopWindow):
    # TODO: spinbox default (if cleaner method is possible)
    #       

    def __init__(self, parent, title, fp, loaded_mat):
        # ==== which file & init
        super(_loadmat, self).__init__(parent, title)
        
        self.loaded_mat = loaded_mat
        self.ret_key = {"data key":"", "event key":"", "sampling rate":0, "nchan":0, "ntimes":0}
        self.srate = tk.IntVar()
        self.nch = tk.IntVar()
        self.ntime = tk.IntVar()
        tk.Label(self, text="Filename: "+fp.split('/')[-1]).grid(row=0, column=0, sticky='w')

        # ==== select key
        spin_val = tuple(reversed([k for k in loaded_mat.keys()]))[:-3] # discard '__globals__', '__version__', '__header__'
        
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
        tk.Entry(self, textvariable=self.srate).grid(row=4, column=1)
        tk.Label(self, text="Channel: ").grid(row=5, column=0, sticky='w')
        tk.Entry(self, textvariable=self.nch).grid(row=5, column=1)
        tk.Label(self, text="Time samples: ").grid(row=6, column=0, sticky='w')
        tk.Entry(self, textvariable=self.ntime).grid(row=6, column=1)

        # ==== confirm
        tk.Button(self, text="Confirm",command=self._key_confirm).grid(row=7, column=0)

    def _shape_view_update(self, var, id, mode): # on spinbox change
        self.data_shape_view.set(str(self.loaded_mat[self.data_key_select.get()].shape)) 
        if self.event_key_select.get() == 'None':
            self.event_shape_view.set('None')
        else:
            self.event_shape_view.set(str(self.loaded_mat[self.event_key_select.get()].shape))
        
    def _key_confirm(self):
        self.ret_key["data key"] =  self.data_key_select.get()
        self.ret_key["event key"] = self.event_key_select.get()
        self.ret_key["sampling rate"] = self.srate.get()
        self.ret_key["nchan"] = self.nch.get()
        self.ret_key["ntimes"] = self.ntime.get()

        assert self.ret_key["data key"] != self.ret_key["event key"], 'Data key and event key should be different.'
        assert self.ret_key["sampling rate"] >0, 'Sampling rate invalid.'
        assert (self.ret_key["nchan"] >0) and (self.ret_key["nchan"] in self.loaded_mat[self.data_key_select.get()].shape), 'Number of channels invalid.'
        assert (self.ret_key["ntimes"] >0) and (self.ret_key["ntimes"] in self.loaded_mat[self.data_key_select.get()].shape), 'Number of time points invalid.'
        self.destroy()

    def _get_result(self):
        return self.ret_key

class LoadMat(LoadTemplate):
    def __init__(self, parent, title):
        super(LoadMat, self).__init__(parent, title)
        self.attr_info = {}

        tk.Button(self, text="Add", command=lambda:self._load_mat()).grid(row=2, column=0, ipadx=3, sticky='w')
        tk.Button(self, text="Confirm", command=lambda:self._confirm_mat()).grid(row=2, column=1, ipadx=3, sticky='w')

    def _load_mat(self):
        selected_tuple = filedialog.askopenfilenames (# +"s" so multiple file selection is available
            parent = self,
            filetypes = (('eeg file', '*.mat'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}

        for fn in selected_tuple:            
            if fn.split('/')[-1] not in self.attr_list.keys():
                selected_data = scipy.io.loadmat(fn)
                attr_info_tmp = {}

                if len(self.data_list) ==0 and self.attr_info=={}:
                    attr_info_tmp = _loadmat(self, "Select Field", fn, selected_data).get_result()
                else:
                    attr_info_tmp = self.attr_info
                
                data_array = selected_data[attr_info_tmp['data key']]
                data_array = self._reshape_array(data_array, attr_info_tmp)
                data_info = mne.create_info(attr_info_tmp['nchan'], attr_info_tmp['sampling rate'], 'eeg')
                
                if self.type_ctrl.get() == 'raw':
                    assert len(data_array.shape) == 2, 'Data dimension invalid.'
                    selected_data = mne.io.RawArray(data_array, data_info)
                else:
                    assert len(data_array.shape) == 3, 'Data dimension invalid.'
                    selected_data = mne.EpochsArray(data_array, data_info)
                new_row = self._make_row(fn, selected_data)
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        # update attr table
        for k,v in attr_list_tmp.items():
            if k not in self.attr_list.keys():
                v_val = tuple([vv.get() for vv in v.values()])
                self.data_attr.insert('', index="end" ,text=k, values=v_val[1:])
                self.attr_list[k] = v
                self.data_list[k] = data_list_tmp[k]
        self.attr_len.set(len(self.attr_list))
        if self.attr_info == {}:
            self.attr_info = attr_info_tmp
        if len(self.attr_list) != 0:
            self.type_raw.config(state="disabled")
            self.type_epoch.config(state="disabled")

    def _reshape_array(self, target, attr_info_tmp):
        # shape for feeding raw constructor: (channel, timepoint)
        # shape for feeding epoch constructor: (epoch, channel, timepoint)
        # {'data key': '', 'event key': '', 'sampling rate': 0, 'nchan': 0, 'ntimes':0}

        target = np.squeeze(target) # squeeze dimension of 1, should ended up be 3d
        target_shape = [s for s in target.shape]
        dim_ch = target_shape.index(attr_info_tmp['nchan'])
        dim_time = target_shape.index(attr_info_tmp['ntimes'])
        if len(target_shape) == 3:
            return np.transpose(target, (3-dim_ch-dim_time, dim_ch, dim_time))
        else:
            return np.transpose(target, (dim_ch, dim_time))
    
    def _confirm_mat(self):
        # check channel number
        check_chs = set(self.data_attr.item(at_row)['values'][3] for at_row in self.data_attr.get_children())
        assert len(check_chs)==1, 'Dataset channel numbers inconsistent.'
        
        ret_attr = {}
        ret_data = {}
        for at_row in self.data_attr.get_children():
            at_val = self.data_attr.item(at_row)['values']
            ret_attr[at_val[0]] = [at_val[1], at_val[1]]
            ret_data[at_val[0]] = self.data_list[at_val[0]]
        if self.type_ctrl.get() == 'raw':
            self.ret_val = Raw(ret_attr, ret_data)
        else:
            self.ret_val = Epochs(ret_attr, ret_data)
        self.destroy()


        




        


