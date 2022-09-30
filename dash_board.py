import tkinter as tk
import tkinter.messagebox
from matplotlib import pyplot as plt
from .load_data import IMPORT_TYPE_MODULE_LIST
from .preprocess import PREPROCESS_MODULE_LIST
from .dataset import DataSplittingSettingWindow
from .training import ModelSelectionWindow, TrainingSettingWindow, TrainingManagerWindow, Trainer
from .evaluation import EVALUATION_MODULE_LIST
from .visualization import PickMontageWindow, VISUALIZATION_MODULE_LIST
from .dataset.data_holder import Epochs
from .dashboard_panel import DatasetPanel, PreprocessPanel, TrainingSchemePanel, TrainingSettingPanel, TrainingStatusPanel
from .base import ValidateException
from copy import deepcopy

class DashBoard(tk.Tk):
    def __init__(self):
        super().__init__()
        # window
        self.child_list = []
        self.minsize(900, 400)
        self.title('Dashboard')
        self.init_menu()
        # panel
        self.columnconfigure([0,1,2], weight=1)
        self.rowconfigure([0,1], weight=1)

        self.dataset_panel = DatasetPanel(self, row=0, column=0)
        self.preprocess_panel = PreprocessPanel(self, row=0, column=1)
        self.training_scheme_panel = TrainingSchemePanel(self, row=0, column=2)
        self.training_setting_panel = TrainingSettingPanel(self, row=1, column=0, columnspan=2)
        self.trainin_status_panel = TrainingStatusPanel(self, row=1, column=2)

        # raw data
        self.loaded_data_list = None
        self.preprocessed_data_list = None
        self.epoch_data = None
        # datasets
        self.datasets = None
        # training
        self.model_holder = None
        self.training_option = None
        self.trainers = None

        self.after(1, self.update_dashboard)
    
    def update_dashboard(self):
        if not self.winfo_exists():
            return

        self.update_idletasks()
        self.dataset_panel.update_panel(self.preprocessed_data_list)
        self.update_idletasks()
        self.preprocess_panel.update_panel(self.preprocessed_data_list)
        self.update_idletasks()
        self.training_scheme_panel.update_panel(self.datasets)
        self.update_idletasks()
        self.training_setting_panel.update_panel(self.model_holder, self.training_option)
        self.update_idletasks()
        self.trainin_status_panel.update_panel(self.trainers)
        self.update_idletasks()

    def init_menu(self):
        menu = tk.Menu(self, tearoff=0)
        # Top level
        import_data_menu = tk.Menu(menu, tearoff=0)
        preprocess_menu = tk.Menu(menu, tearoff=0)
        training_menu = tk.Menu(menu, tearoff=0)
        evaluation_menu = tk.Menu(menu, tearoff=0)
        visualization_menu = tk.Menu(menu, tearoff=0)

        menu.add_cascade(label="Import data", menu=import_data_menu)
        menu.add_cascade(label="Preprocess", menu=preprocess_menu)
        menu.add_cascade(label="Training", menu=training_menu)
        menu.add_cascade(label="Evaluation", menu=evaluation_menu)
        menu.add_cascade(label="Visualization", menu=visualization_menu)
        
        # import data
        for import_module in IMPORT_TYPE_MODULE_LIST:
            import_data_menu.add_command(label=import_module.command_label, command=lambda var=import_module:self.import_data(var))
        
        # preprocess/epoching
        for preprocess_module in PREPROCESS_MODULE_LIST:
            preprocess_menu.add_command(label=preprocess_module.command_label, command=lambda var=preprocess_module:self.preprocess(var))
        preprocess_menu.add_command(label='Reset', command=self.reset_preprocess)
        
        # training
        training_menu.add_command(label='Dataset Splitting', command=self.split_data)
        training_menu.add_command(label='Model Selection', command=self.select_model)
        training_menu.add_command(label='Training Setting', command=self.training_setting)
        training_menu.add_command(label='Generate Training Plan', command=self.generate_plan)
        training_menu.add_command(label='Training Manager', command=self.open_training_manager)

        # evaluation
        for evaluation_module in EVALUATION_MODULE_LIST:
            evaluation_menu.add_command(label=evaluation_module.command_label, command=lambda var=evaluation_module:self.evaluate(var))
        
        # visualization
        visualization_menu.add_command(label='Set Montage', command=lambda:self.set_montage())
        for visualization_module in VISUALIZATION_MODULE_LIST:
            visualization_menu.add_command(label=visualization_module.command_label, command=lambda var=visualization_module:self.visualize(var))

        self.config(menu=menu)
    
    def warn_flow_cleaning(self):
        if tk.messagebox.askokcancel(parent=self, title='Warning', message='This step has already been done, all following data will be removed if you reset this step.\nDo you want to proceed?'):
            return True
        return False

    # data
    def set_preprocessed_data_list(self, preprocessed_data_list):
        self.preprocessed_data_list = preprocessed_data_list
        if not preprocessed_data_list[0].is_raw():
            self.epoch_data = Epochs(preprocessed_data_list)
        else:
            self.epoch_data = None

    def import_data(self, import_module):
        if self.preprocessed_data_list:
            if not self.warn_flow_cleaning():
                return
        loaded_data_list = import_module(self).get_result()
        if loaded_data_list:
            self.loaded_data_list = loaded_data_list
            self.set_preprocessed_data_list(deepcopy(loaded_data_list))
            self.clean_datasets()
            self.update_dashboard()

    def preprocess(self, preprocess_module):
        if self.datasets:
            if not self.warn_flow_cleaning():
                return
        preprocessed_data_list = preprocess_module(self, self.preprocessed_data_list).get_result()
        if preprocessed_data_list:
            self.set_preprocessed_data_list(preprocessed_data_list)
            self.clean_datasets()
            self.update_dashboard()

    def reset_preprocess(self):
        if self.datasets:
            if not self.warn_flow_cleaning():
                return
        if self.loaded_data_list:
            self.set_preprocessed_data_list(deepcopy(self.loaded_data_list))
            tk.messagebox.showinfo(parent=self, title='Success', message='OK')
        else:
            raise ValidateException(window=self, message='No valid data has been loaded')
        self.clean_datasets()
        self.update_dashboard()
    
    # train
    def split_data(self):
        if self.trainers:
            if not self.warn_flow_cleaning():
                return
        datasets = DataSplittingSettingWindow(self, self.epoch_data).get_result()
        if datasets:
            self.datasets = datasets
            self.clean_trainer()
            self.update_dashboard()

    def select_model(self):
        if self.trainers:
            if not self.warn_flow_cleaning():
                return
        model_holder = ModelSelectionWindow(self).get_result()
        if model_holder:
            self.model_holder = model_holder
            self.clean_trainer()
            self.update_dashboard()

    def training_setting(self):
        if self.trainers:
            if not self.warn_flow_cleaning():
                return
        training_option = TrainingSettingWindow(self).get_result()
        if training_option:
            self.training_option = training_option
            self.clean_trainer()
            self.update_dashboard()

    def generate_plan(self):
        if self.trainers:
            if not self.warn_flow_cleaning():
                return
        if not self.datasets:
            raise ValidateException(window=self, message='No valid dataset is generated')

        trainers = []
        option = self.training_option
        model_holder = self.model_holder
        datasets = self.datasets
        for dataset in datasets:
            trainers.append(Trainer(model_holder, dataset, option))
        self.trainers = trainers
        self.open_training_manager()
        self.update_dashboard()

    def open_training_manager(self):
        TrainingManagerWindow(self, self.trainers)

    # eval
    def evaluate(self, evaluation_module):
        evaluation_module(self, self.trainers)
    
    # visualize
    def set_montage(self):
        if type(self.epoch_data) != Epochs:
            raise ValidateException(window=self, message='No valid epoch data is generated')
            return
        chs, positions = PickMontageWindow(self, self.preprocessed_data_list.get_channel_names()).get_result()
        if chs is not None and positions is not None:
            self.preprocessed_data_list.set_channels(chs, positions)
            self.update_dashboard()

    def visualize(self, visualization_module):
        visualization_module(self, self.trainers)

    # destroy
    def append_child_window(self, child):
        self.child_list.append(child)
    
    def remove_child_window(self, child):
        self.child_list.remove(child)
       
    def check_training(self):
        if TrainingManagerWindow.task:
            if tk.messagebox.askokcancel(parent=self, title='Warning', message='Training is in progress.\nDo you want to exit?'):
                TrainingManagerWindow.task.set_interrupt()
                return True
            else:
                return False
        return True

    def destroy(self):
        if not self.check_training():
            return
        
        child_list = self.child_list.copy()
        for child in child_list:
            if child.destroy():
                return True
        
        plt.close('all')
        # recycling
        print('recycling...')
        self.withdraw()
        self.after(1000, super().destroy)

    # clean work flow

    def clean_datasets(self):
        self.datasets = None
        self.clean_trainer()

    def clean_trainer(self):
        self.trainers = None

