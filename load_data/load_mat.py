from .load_template import *
from .load_template import _loadmat

class LoadMat(LoadTemplate):
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
        tk.Button(self.stat_frame, text="Edit selected keys", command=lambda:self._clear_key()).grid(row=5, column=0)

    def _load(self):
        selected_tuple = filedialog.askopenfilenames (
            parent = self,
            filetypes = (('.mat files', '*.mat'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}
        raw_event_tmp = {}
        raw_event_src_tmp = {}
        attr_info_tmp = self.attr_info

        for fn in selected_tuple:            
            if fn.split('/')[-1] not in self.attr_list.keys():
                selected_data = scipy.io.loadmat(fn)
                if (self.attr_info['nchan'] ==0 and attr_info_tmp['nchan']==0 ) \
                    or any(k in selected_data.keys() for k in attr_info_tmp['data key'])==False \
                    or any(k in selected_data.keys() for k in attr_info_tmp['event key'])==False:
                    attr_info_tmp = _loadmat(self, "Select Field", fn,attr_info_tmp, selected_data).get_result()
                if sum(k in attr_info_tmp['data key'] for k in selected_data.keys())>1:
                    raise ValidateException(window=self, message='Data has multiple keys identified as containing data.')
                for k in attr_info_tmp['data key']:
                    if k in selected_data.keys():
                        data_array = selected_data[k]
                        break
                data_array = self._reshape_array(data_array, attr_info_tmp)
                if len(data_array.shape)==2:
                    attr_info_tmp['nchan'] = data_array.shape[0]
                    attr_info_tmp['sfreq'] = data_array.shape[1]

                data_info = mne.create_info(attr_info_tmp['nchan'], attr_info_tmp['sfreq'], 'eeg')
                selected_data, new_event = self._data_from_array(data_array, data_info, attr_info_tmp, selected_data)
                if new_event != None:
                    raw_event_tmp[fn.split('/')[-1]] = new_event
                    raw_event_src_tmp[fn.split('/')[-1]] = 3
                      
                new_row, _ , __ = self._make_row(fn, selected_data)

                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data

        self._list_update(attr_list_tmp, data_list_tmp, raw_event_tmp, raw_event_src_tmp)
        if self.nch.get()==0:
            self.attr_info = attr_info_tmp
            self.nch.set(self.attr_info['nchan'])
            self.srate.set(self.attr_info['sfreq'])