from tkinter import CENTER
from turtle import heading, width
from Template import *

# TODO: consistency of multiple files
#       event

class EditRow(TopWindow):
    def __init__(self, parent, title, header, target):
        super(EditRow, self).__init__(parent, title)
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

class LoadSet(TopWindow):
    def __init__(self, parent, title):
        # TODO: hover cursor shows full filepath msg

        # ==== initialize ==== 
        super(LoadSet, self).__init__(parent, title)
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


        tk.Button(self, text="Add", command=lambda:self._load_set()).grid(row=2, column=0, ipadx=3, sticky='w')
        tk.Button(self, text="Confirm", command=lambda:self._confirm_set()).grid(row=2, column=1, ipadx=3, sticky='w')
    
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
        del_row = EditRow(self, "Edit data attribute", self.attr_header, self.attr_list[selected_row['text']]).get_result()
        self.data_attr.delete(*self.data_attr.get_children())
        if del_row:
            self.attr_list.pop(selected_row['text'])
            self.data_list.pop(selected_row['text'])
        
        for k,v in self.attr_list.items():
            v_val = tuple([vv.get() for vv in v.values()])
            self.data_attr.insert('', index="end" ,text=k, values=v_val[1:])

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

    def _get_result(self):
        return self.ret_val



class LoadMat(TopWindow):
    pass