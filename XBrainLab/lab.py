import traceback
from copy import deepcopy
from .load_data import RawDataLoader, Raw
from .dataset import DataSplittingConfig, Dataset
from .dataset import Epochs, DatasetGenerator
from .training import TrainingOption, ModelHolder 
from .training import TrainingPlanHolder, Trainer
from .utils import validate_type, validate_list_type, validate_issubclass

from .preprocessor.base import PreprocessBase

class XBrainLab:

    def __init__(self):
        self.ui = None
        # raw data
        self.loaded_data_list = None
        self.preprocessed_data_list = None
        self.epoch_data = None
        # datasets
        self.datasets = None
        # training
        self.model_holder = None
        self.training_option = None
        self.trainer = None
        #
        self.script_history = None

    # load data
    def get_raw_data_loader(self):
        return RawDataLoader()

    def set_loaded_data_list(self, loaded_data_list, force_update=False):
        validate_list_type(loaded_data_list, Raw, 'loaded_data_list')
        self.set_preprocessed_data_list(preprocessed_data_list=deepcopy(loaded_data_list), force_update=force_update)
        self.loaded_data_list = loaded_data_list

    # preprocess
    def set_preprocessed_data_list(self, preprocessed_data_list, force_update=False):
        validate_list_type(preprocessed_data_list, Raw, 'preprocessed_data_list')
        self.clean_datasets(force_update=force_update)
        
        self.preprocessed_data_list = preprocessed_data_list
        for preprocessed_data in preprocessed_data_list:
            if preprocessed_data_list[0].is_raw():
                self.epoch_data = None
                return
        self.epoch_data = Epochs(preprocessed_data_list)

    def reset_preprocess(self, force_update=False):
        if self.loaded_data_list:
            self.set_preprocessed_data_list(deepcopy(self.loaded_data_list), force_update=force_update)
        else:
            raise ValueError('No valid data has been loaded')

    def preprocess(self, preprocessor, **kargs):
        validate_issubclass(preprocessor, PreprocessBase, 'preprocessor')
        preprocessor = preprocessor(self.preprocessed_data_list)
        preprocessor.check_data()
        preprocessed_data_list = preprocessor.data_preprocess(**kargs)
        self.set_preprocessed_data_list(preprocessed_data_list)
    
    # split data
    def get_datasets_generator(self, config):
        validate_type(config, DataSplittingConfig ,"config")
        return DatasetGenerator(self.epoch_data, config)

    def set_datasets(self, datasets, force_update=False):
        validate_list_type(datasets, Dataset, 'datasets')
        self.clean_trainer(force_update=force_update)
        self.datasets = datasets

    def set_training_option(self, training_option, force_update=False):
        validate_type(training_option, TrainingOption,'training_option')
        self.clean_trainer(force_update=force_update)
        self.training_option = training_option

    # traing
    def set_model_holder(self, model_holder, force_update=False):
        validate_type(model_holder, ModelHolder, 'model_holder')
        self.clean_trainer(force_update=force_update)
        self.model_holder = model_holder

    def generate_plan(self, force_update=False):
        self.clean_trainer(force_update=force_update)
        
        if not self.datasets:
            raise ValueError('No valid dataset is generated')
        if not self.training_option:
            raise ValueError('No valid training option is generated')
        if not self.model_holder:
            raise ValueError('No valid model holder is generated')
        
        training_plan_holders = []
        option = self.training_option
        model_holder = self.model_holder
        datasets = self.datasets
        for dataset in datasets:
            training_plan_holders.append(TrainingPlanHolder(model_holder, dataset, option))
        self.trainer = Trainer(training_plan_holders)

    def train(self, interact=False):
        if not self.trainer:
            raise ValueError('No valid trainer is generated')
        
        self.trainer.run(interact=interact)
    
    def stop_training(self):
        if self.trainer:
            return self.trainer.set_interrupt()

    def is_training(self):
        if self.trainer:
            return self.trainer.is_running()
        return False

    # vis
    def set_channels(self, chs, positions):
        self.epoch_data.set_channels(chs, positions)

    # eval
    def export_output_csv(self, filepath, plan_name, real_plan_name):
        training_plan_holders = None
        if self.trainer:
            training_plan_holders = self.trainer.get_training_plan_holders()
        if not training_plan_holders:
            raise ValueError("No valid training plan is generated")
        record = None
        for training_plan_holder in training_plan_holders:
            if training_plan_holder.get_name() == plan_name:
                for plan in training_plan_holder.get_plans():
                    if plan.get_name() == real_plan_name:
                        record = plan.get_eval_record()
                        if not record:
                            raise ValueError('No evaluation record for this training plan')
                        record.export_csv(filepath)
                        return
                raise ValueError(f'No real plan named "{real_plan_name}" for "{plan_name}"')
        raise ValueError(f'No training plan named "{plan_name}"')

    # clean work flow
    def should_clean_raw_data(self, interact=True):
        response = self.loaded_data_list is not None or self.should_clean_datasets(interact)
        if response and interact:
            raise ValueError('This step has already been done, all following data will be removed if you reset this step.\nPlease clean_raw_data first.')
        return response

    def clean_raw_data(self, force_update=False):
        self.clean_datasets(force_update=force_update)
        if not force_update:
            self.should_clean_raw_data(interact=True)
        self.loaded_data_list = None
        self.preprocessed_data_list = None
        self.epoch_data = None

    def should_clean_datasets(self, interact=True):
        response = self.datasets is not None or self.should_clean_trainer(interact)
        if response and interact:
            raise ValueError('This step has already been done, all following data will be removed if you reset this step.\nPlease clean_datasets first.')
        return response

    def clean_datasets(self, force_update=False):
        self.clean_trainer(force_update=force_update)
        if not force_update:
            self.should_clean_datasets(interact=True)
        self.datasets = None

    def should_clean_trainer(self, interact=True):
        response = self.trainer is not None
        if response and interact:
            raise ValueError('This step has already been done, all following data will be removed if you reset this step.\nPlease clean_trainer first.')
        return response

    def clean_trainer(self, force_update=False):
        if not force_update:
            self.should_clean_trainer(interact=True)
        if self.trainer:
            self.trainer.clean(force_update=force_update)
        self.trainer = None
    #
    def get_script(self):
        return self.script_history

    def set_script(self, script_history):
        self.script_history = script_history

    def show_ui(self, interact=False):
        try:
            if(self.ui):
                self.ui.destroy(force=True)
        except Exception as e:
            pass
        self.ui = None 
        try:
            from .ui.dash_board import DashBoard
            self.ui = DashBoard(study=self)
            if not interact:
                self.ui_loop()
            return self.ui
        except Exception as e:
            traceback.print_exc()
            try:
                self.ui.destroy(force=True)
            except:
                pass
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showerror(parent=root, title='Error', message=e)
                root.destroy()
            except Exception as e:
                traceback.print_exc()
                raise e
    
    def ui_loop(self):
        if self.ui is None or not self.ui.window_exist:
            self.show_ui(interact=False)
        else:
            self.ui.mainloop()
            self.ui = None

    def ui_func_wrapper(func):
        def wrap(*args, **kwargs):
            win = args[0]
            trainers = None
            if win.trainer:
                trainers = win.trainer.get_training_plan_holders()
            if not win.ui:
                win.show_ui(interact=True)
            func(*args, **kwargs, trainers=trainers)
            win.ui_loop()

        return wrap

    @ui_func_wrapper
    def show_plot(self, plot_type, plan_name, real_plan_name, trainers):
        from .ui.widget import PlotFigureWindow
        PlotFigureWindow(parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name)

    @ui_func_wrapper
    def show_grad_plot(self, plot_type, plan_name, real_plan_name, absolute, trainers):
        from .ui.visualization import PlotABSFigureWindow
        PlotABSFigureWindow(parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name, absolute=absolute)
    
    @ui_func_wrapper
    def show_grad_topo_plot(self, plot_type, plan_name, real_plan_name, absolute, trainers):
        from .ui.visualization import PlotTopoABSFigureWindow
        PlotTopoABSFigureWindow(parent=self.ui, trainers=trainers,
            plot_type=plot_type, plan_name=plan_name, real_plan_name=real_plan_name, absolute=absolute)

    @ui_func_wrapper
    def show_performance(self, metric, trainers):
        from .ui.evaluation import EvaluationTableWindow
        EvaluationTableWindow(self.ui, trainers, metric)
    