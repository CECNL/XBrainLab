from Template import *
from ImportData import *
from Epoch import *
from Preprocess import *
class _editevent(TopWindow):
    # TODO: layout
    def __init__(self, parent, title, old_event):
        super(_editevent, self).__init__(parent, title)
        self.new_event = {}

        self.new_event_name = {i:tk.StringVar() for i in range(len(old_event))}
        self.new_event_id = {i:tk.IntVar() for i in range(len(old_event))}
        tk.Label(self, text="Event id: ").grid()
        i = 0
        for k,v in old_event.items():
            self.new_event_name[i].set(k)
            self.new_event_id[i].set(int(v))
            tk.Entry(self, textvariable=self.new_event_name[i]).grid(row=i+1, column=0)
            tk.Entry(self, textvariable=self.new_event_id[i]).grid(row=i+1, column=1)
            i += 1
        tk.Button(self, text="Cancel", command=lambda:self._confirm(0, len(old_event))).grid(row=i+1, column=0)
        tk.Button(self, text="Confirm", command=lambda:self._confirm(1, len(old_event))).grid(row=i+1, column=1)
    
    def _confirm(self, conf=0, event_len=0):
        if conf ==1:
            for i in range(event_len):
                assert self.new_event_id[i].get() not in self.new_event.values(), 'Duplicate event id.'
                assert self.new_event_name[i].get() not in self.new_event.keys(), 'Duplicate event name.'
                self.new_event[self.new_event_name[i].get()] = self.new_event_id[i].get()
        
        self.destroy()
    def _get_result(self):
        return self.new_event

class DashBoard(TopWindow):
    def __init__(self, parent):
        self.DashBoard = parent
        self.DashBoard.geometry("500x500")
        self.DashBoard.title("Exbrainable")
        
        self.loaded_data = None
        self.nav_bar = tk.Menu(self.DashBoard, tearoff=0)
        self.init_Menu()
        self.DashBoard.config(menu=self.nav_bar)

    def init_Menu(self):
        # Navigation bar: file, dataset, model, training, help
        # TODO: menu state disable

        file_type_menu = tk.Menu(self.nav_bar, tearoff=0)
        file_type_menu.add_command(label="Import SET file (EEGLAB toolbox)", command=lambda:self._import_data('set'))
        file_type_menu.add_command(label="Import EDF/EDF+ file (BIOSIG toolbox)", command=lambda:self._import_data('edf'))
        file_type_menu.add_command(label="Import CNT file (Neuroscan)", command=lambda:self._import_data('cnt'))
        file_type_menu.add_command(label="Import MAT file (Matlab array)", command=lambda:self._import_data('mat'))
        file_type_menu.add_command(label="Import NPY/NPZ file (Numpy array)", command=lambda:self._import_data('np'))


        file_menu = tk.Menu(self.nav_bar,tearoff=0)
        file_menu.add_cascade(label="Import data",menu = file_type_menu)
        self.nav_bar.add_cascade(label="File", menu=file_menu)


        self.data_menu = tk.Menu(self.nav_bar,tearoff=0)
        self.data_menu.add_command(label="Edit event id", command=lambda:self._edit_event_id())
        self.data_menu.entryconfig("Edit event id", state="disabled")
        self.nav_bar.add_cascade(label="Dataset",menu = self.data_menu)

        data_menu = tk.Menu(self.nav_bar, tearoff=0)
        data_menu.add_command(label="Channel", command=lambda: self._preprocess_channel())
        data_menu.add_command(label="Filtering", command=lambda: self._preprocess_filtering())
        data_menu.add_command(label="Resample", command=lambda: self._preprocess_resample())
        self.nav_bar.add_cascade(label="Preprocess", menu=data_menu)

        data_menu = tk.Menu(self.nav_bar, tearoff=0)
        data_menu.add_command(label="Time", command=lambda: self._epoching_time())
        data_menu.add_command(label="Window", command=lambda: self._epoching_window())
        data_menu.add_command(label="Baseline Removal", command=lambda: self._epoching_baselineRemoval())
        self.nav_bar.add_cascade(label="Epoching", menu=data_menu)

        model_menu = tk.Menu(self.nav_bar,tearoff=0)
        self.nav_bar.add_cascade(label="Model",menu = model_menu)

        train_menu = tk.Menu(self.nav_bar,tearoff=0)
        self.nav_bar.add_cascade(label="Train",menu = train_menu)

        help_menu = tk.Menu(self.nav_bar, tearoff=0)
        self.nav_bar.add_cascade(label="Help", menu=help_menu)

    def _import_data(self, type_key="set"):
        load_title = "Load .{} file".format(type_key)
        load_map = {
            'set': lambda s, lt:LoadSet(s, lt).get_result(),
            'edf': lambda s, lt:LoadEdf(s, lt).get_result(),
            'cnt': lambda s, lt:LoadCnt(s, lt).get_result(),
            'mat': lambda s, lt:LoadMat(s, lt).get_result(),
            'np': lambda s, lt:LoadNp(s, lt).get_result()

        }
        self.loaded_data = load_map[type_key](self, load_title)
        if isinstance(self.loaded_data, Raw): # raw with event
            if self.loaded_data.event_id_map != {}:
                self.data_menu.entryconfig("Edit event id", state="normal")
        else: # 
            self.data_menu.entryconfig("Edit event id", state="normal")

        self.loaded_data.inspect()
    def _edit_event_id(self):
        new_event = _editevent(self, "Edit event", self.loaded_data.event_id).get_result()
        
        # event id update
        old_event = self.loaded_data.event_id
        event_map = {k:v for k,v in zip(old_event.values(), new_event.values())}
        self.loaded_data.event_id = new_event

        # label update
        if isinstance(self.loaded_data, Raw):
            for i in range(len(self.loaded_data.label)):
                for j in range(self.loaded_data.label[i].shape[0]):
                    self.loaded_data.label[i][j] = event_map[self.loaded_data.label[i][j]]
        else:
            for fn in self.loaded_data.id_map.keys():
                self.loaded_data.data[self.loaded_data.id_map[fn]].event_id = new_event
                old_label = self.loaded_data.data[self.loaded_data.id_map[fn]].events[:,2]
                new_label = np.copy(old_label)
                for k,v in event_map.items(): new_label[old_label==k] = v
                self.loaded_data.data[self.loaded_data.id_map[fn]].events[:,2] = new_label

    def _preprocess_channel(self):
        self.loaded_data = Channel(self, "Channel").get_result()
        self.loaded_data.inspect()

    def _preprocess_filtering(self):
        self.loaded_data = Filtering(self, "Filtering").get_result()
        self.loaded_data.inspect()

    def _preprocess_resample(self):
        self.loaded_data = Resample(self, "Resample").get_result()
        self.loaded_data.inspect()

    def _epoching_time(self):
        self.loaded_data = TimeEpoch(self, "Time").get_result()
        self.loaded_data.inspect()

    def _epoching_window(self):
        self.loaded_data = WindowEpoch(self, "Window").get_result()
        self.loaded_data.inspect()

    def _epoching_baselineRemoval(self):
        self.loaded_data = BaselineRemoval(self, "Baseline Removal").get_result()
        self.loaded_data.inspect()
        