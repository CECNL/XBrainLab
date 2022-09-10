from select import select
from tabnanny import verbose
from Template import *
from ImportData import *

# layout 等東西都加完再喬 ?

class DashBoard(TopWindow):
    def __init__(self, parent):
        self.DashBoard = parent
        self.DashBoard.geometry("500x500")
        self.DashBoard.title("Exbrainable")
        
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

        data_menu = tk.Menu(self.nav_bar,tearoff=0)
        self.nav_bar.add_cascade(label="Dataset",menu = data_menu)

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
        win = load_map[type_key](self, load_title)
        win.inspect()
        