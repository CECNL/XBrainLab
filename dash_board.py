import tkinter as tk
from .load_data import LoadSet, LoadEdf, LoadCnt, LoadMat, LoadNp
from .preprocess import Channel, Filtering, Resample, TimeEpoch, WindowEpoch, EditEvent
from .dataset import DataSplittingSettingWindow
from .training import ModelSelectionWindow, TrainingSettingWindow, TrainingManagerWindow , TrainingPlan
from .evaluation import ConfusionMatrixWindow, EvaluationTableWindow, ModelOutputWindow

class DashBoard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("500x500")
        self.title('Dashboard')
        self.init_menu()
        # raw data
        self.loaded_data = None
        self.preprocessed_data = None
        self.preprocess_history = []
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
            import_data_menu.add_command(label=import_data_type.command_label, command=lambda var=import_data_type:self.import_data(var))
        file_menu.add_cascade(label="Import data", menu=import_data_menu)
        
        # preprocess/epoching
        preprocess_type_list = [Channel, Filtering, Resample, EditEvent, TimeEpoch, WindowEpoch]
        for preprocess_type in preprocess_type_list:
            preprocess_menu.add_command(label=preprocess_type.command_label, command=lambda var=preprocess_type:self.preprocess(var))
        preprocess_menu.add_command(label='Reset', command=self.reset_preprocess)
        
        # training
        training_menu.add_command(label='Dataset Splitting', command=self.split_data)
        training_menu.add_command(label='Model Selection', command=self.select_model)
        training_menu.add_command(label='Training Setting', command=self.training_setting)
        training_menu.add_command(label='Generate Training Plan', command=self.generate_plan)
        training_menu.add_command(label='Training Manager', command=self.open_training_manager)
        # evaluation
        evaluation_type_list = [ConfusionMatrixWindow, EvaluationTableWindow, ModelOutputWindow]
        for evaluate_type in evaluation_type_list:
            evaluation_menu.add_command(label=evaluate_type.command_label, command=lambda var=evaluate_type:self.evaluate(var))
        
        self.config(menu=menu)

    def warn_flow_cleaning(self):
        if tk.messagebox.askokcancel(parent=self, title='Warning', message='This step has already been done, all following data will be removed if you reset this step.\nDo you want to proceed?'):
            return True
        return False
    # data
    def import_data(self, import_data_type):
        if len(self.preprocess_history) > 0 or self.datasets:
            if not self.warn_flow_cleaning():
                return
        loaded_data = import_data_type(self).get_result()
        if loaded_data:
            self.loaded_data = loaded_data
            self.preprocessed_data = loaded_data.copy()
            # TODO clear working flow

    def preprocess(self, preprocess_type):
        if self.datasets:
            if not self.warn_flow_cleaning():
                return
        preprocessed_data, log = preprocess_type(self, self.preprocessed_data).get_result()
        if preprocessed_data and log:
            self.preprocessed_data = preprocessed_data
            self.preprocess_history.append(log)
            # TODO clear working flow

    def reset_preprocess(self):
        if self.datasets:
            if not self.warn_flow_cleaning():
                return
        if self.loaded_data:
            self.preprocessed_data = self.loaded_data.copy()
            tk.messagebox.showinfo(parent=self, title='Success', message='OK')
        else:
            tk.messagebox.showerror(parent=self, title='Error', message='No valid data is loaded')
        # TODO clear working flow
    # train
    def split_data(self):
        if self.training_plan_holders:
            if not self.warn_flow_cleaning():
                return
        datasets = DataSplittingSettingWindow(self, self.preprocessed_data).get_result()
        if datasets:
            self.datasets = datasets
        # TODO clear working flow

    def select_model(self):
        if self.training_plan_holders:
            if not self.warn_flow_cleaning():
                return
        model_holder = ModelSelectionWindow(self).get_result()
        if model_holder:
            self.model_holder = model_holder
        # TODO clear working flow

    def training_setting(self):
        if self.training_plan_holders:
            if not self.warn_flow_cleaning():
                return
        training_option = TrainingSettingWindow(self).get_result()
        if training_option:
            self.training_option = training_option
        # TODO clear working flow

    def generate_plan(self):
        if self.training_plan_holders:
            if not self.warn_flow_cleaning():
                return
        if not self.datasets:
            tk.messagebox.showerror(parent=self, title='Error', message='No valid dataset is generated')
            return
        if not self.training_option:
            tk.messagebox.showerror(parent=self, title='Error', message='No valid training option is generated')
            return
        if not self.model_holder:
            tk.messagebox.showerror(parent=self, title='Error', message='No valid model is selected')
            return

        training_plan_holders = []
        option = self.training_option
        model_holder = self.model_holder
        datasets = self.datasets
        for dataset in datasets:
            training_plan_holders.append(TrainingPlan(option, model_holder, dataset))
        self.training_plan_holders = training_plan_holders
        self.open_training_manager()
        # TODO clear working flow

    def open_training_manager(self):
        TrainingManagerWindow(self, self.training_plan_holders)

    # eval
    def evaluate(self, evaluation_type):
        evaluation_type(self, self.training_plan_holders)

    # destroy
    def check_training(self):
        from .training.training_manager import TrainingManagerWindow
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
        super().destroy()
        from matplotlib import pyplot as plt
        plt.close('all')

    # clean work flow

