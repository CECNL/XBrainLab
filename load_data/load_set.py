from .load_template import *

class LoadSet(LoadTemplate):
    command_label = "Import SET file (EEGLAB toolbox)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .set files")
   
    def _load(self):
        selected_tuple = filedialog.askopenfilenames (
            parent = self,
            filetypes = (('.set files', '*.set'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}
        raw_event_tmp = {}
        raw_event_src_tmp = {}

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys():
                if self.type_ctrl.get() == 'raw':
                    try:
                        selected_data = mne.io.read_raw_eeglab(fn, uint16_codec='latin1', preload=True)
                    except (TypeError):
                        tk.messagebox.showwarning(parent=self, title="Warning", message="Detected multiple epochs, switch to epochs loading")
                        self.type_ctrl.set('epochs')
                        selected_data = mne.io.read_epochs_eeglab(fn, uint16_codec='latin1')
                else:
                    try:
                        selected_data = mne.io.read_epochs_eeglab(fn, uint16_codec='latin1')
                    except (ValueError):
                        tk.messagebox.showwarning(parent=self, title="Warning", message="Detected number of trial < 2, switch to epochs loading")
                        self.type_ctrl.set('raw')
                        selected_data = mne.io.read_raw_eeglab(fn, uint16_codec='latin1', preload=True)
                    
                new_row, new_event, new_event_src = self._make_row(fn, selected_data)
                if new_event != None and new_event_src>=0:
                    raw_event_tmp[fn.split('/')[-1]] = new_event
                    raw_event_src_tmp[fn.split('/')[-1]] = new_event_src
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        self._list_update(attr_list_tmp, data_list_tmp, raw_event_tmp, raw_event_src_tmp)
