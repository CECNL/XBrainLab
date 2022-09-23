from .load_template import *

class LoadCnt(LoadTemplate):
    # TODO: untested, lack of test data
    # combine with load edf???
    command_label = "Import CNT file (Neuroscan)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .cnt files")
        self.type_raw.config(state="disabled")# only supporting raw
        self.type_epoch.config(state="disabled")

    def _load(self):
        selected_tuple = filedialog.askopenfilenames (
            parent = self,
            filetypes = (('.cnt files', '*.cnt'),)
        )
        attr_list_tmp = {}
        data_list_tmp = {}
        raw_event_tmp = {}
        raw_event_src_tmp = {}

        for fn in selected_tuple:
            if fn.split('/')[-1] not in self.attr_list.keys(): 
                selected_data = mne.io.read_raw_cnt(fn, preload=True)
                new_row, new_event, new_event_src = self._make_row(fn, selected_data)
                if new_event != None and new_event_src>=0:
                    raw_event_tmp[fn.split('/')[-1]] = new_event
                    raw_event_src_tmp[fn.split('/')[-1]] = new_event_src
                attr_list_tmp[fn.split('/')[-1]] = new_row
                data_list_tmp[fn.split('/')[-1]] = selected_data
        
        self._list_update(attr_list_tmp, data_list_tmp, raw_event_tmp, raw_event_src_tmp)

