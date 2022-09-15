import tkinter as tk
from .load_data import LoadSet, LoadEdf, LoadCnt, LoadMat, LoadNp
from .preprocess import Channel, Filtering, Resample, TimeEpoch, WindowEpoch, EditEvent
from .dataset import DataSplittingSettingWindow
from .training import ModelSelectionWindow, TrainingSettingWindow, TrainingManagerWindow 

class DashBoard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("500x500")
        self.title('Dashboard')
        self.init_menu()
        # raw data
        self.loaded_data = None
        self.preprocessed_data = None
        # datasets
        self.datasets = None
        # training
        self.model_holder = None
        self.training_option = None
        self.training_plan_holders = None

    def init_menu(self):
        menu = tk.Menu(self, tearoff=0)
        # Top level
        file_menu = tk.Menu(menu, tearoff=0)
        preprocess_menu = tk.Menu(menu, tearoff=0)
        training_menu = tk.Menu(menu, tearoff=0)
        evaluation_menu = tk.Menu(menu, tearoff=0)

        menu.add_cascade(label="File", menu=file_menu)
        menu.add_cascade(label="Preprocess", menu=preprocess_menu)
        menu.add_cascade(label="Training", menu=training_menu)
        menu.add_cascade(label="Evaluation", menu=evaluation_menu)
        
        # file_menu
        import_data_menu = tk.Menu(menu, tearoff=0)
        import_data_type_list = [LoadSet, LoadEdf, LoadCnt, LoadMat, LoadNp]
        for import_data_type in import_data_type_list:
            import_data_menu.add_command(label=import_data_type.command_label, command=lambda:self.import_data(import_data_type))
        file_menu.add_cascade(label="Import data", menu=import_data_menu)
        
        # preprocess/epoching
        preprocess_type_list = [Channel, Filtering, Resample, EditEvent, TimeEpoch, WindowEpoch]
        for preprocess_type in preprocess_type_list:
            preprocess_menu.add_command(label=preprocess_type.command_label, command=lambda:self.preprocess(preprocess_type))
        preprocess_menu.add_command(label='Reset', command=self.reset_preprocess)
        
        # training
        training_menu.add_command(label='Dataset Splitting', command=self.split_data)
        training_menu.add_command(label='Model Selection', command=self.select_model)
        training_menu.add_command(label='Training setting', command=self.training_setting)
        training_menu.add_command(label='Training Manager', command=self.open_training_manager)
        # evaluation

        self.config(menu=menu)


    def import_data(self, import_data_type):
        # TODO warning resetting flow
        loaded_data = import_data_type(self)
        if loaded_data:
            self.loaded_data = loaded_data
            self.preprocessed_data = loaded_data

    def preprocess(self, preprocess_type):
        # TODO check if data is loaded
        # TODO warning resetting flow
        loaded_data = preprocess_type(self, self.preprocessed_data)
        if loaded_data:
            self.preprocessed_data = loaded_data

    def reset_preprocess(self):
        # TODO warning resetting flow
        pass

    def split_data(self):
        # TODO check preprocessed_data is epoched
        datasets = DataSplittingSettingWindow(self, self.preprocessed_data).get_result()
        if datasets:
            self.datasets = datasets

    def select_model(self):
        # TODO warning resetting flow
        model_holder = ModelSelectionWindow(self).get_result()
        if model_holder:
            self.model_holder = model_holder

    def training_setting(self):
        # TODO warning resetting flow
        training_option = TrainingSettingWindow(self).get_result()
        if training_option:
            self.training_option = training_option

    def open_training_manager(self):
        # TODO check training_plan_holders is ready
        TrainingManagerWindow(self, self.training_plan_holders)
