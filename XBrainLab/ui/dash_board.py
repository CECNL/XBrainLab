import tkinter as tk
import tkinter.messagebox
from copy import deepcopy

from matplotlib import pyplot as plt

from XBrainLab.dataset import Epochs

from .base import ValidateException
from .dashboard_panel import (
    DatasetPanel,
    PreprocessPanel,
    TrainingSchemePanel,
    TrainingSettingPanel,
    TrainingStatusPanel,
    ExplanationSettingPanel
)
from .dataset import DataSplittingSettingWindow
from .evaluation import EVALUATION_MODULE_LIST
from .load_data import IMPORT_TYPE_MODULE_LIST
from .preprocess import PREPROCESS_MODULE_LIST
from .script import Script, ScriptPreview, ScriptType
from .training import (
    ModelSelectionWindow,
    TestOnlySettingWindow,
    TrainingManagerWindow,
    TrainingSettingWindow,
)
from .visualization import VISUALIZATION_MODULE_LIST, PickMontageWindow, SetSaliencyWindow


class DashBoard(tk.Tk):
    def __init__(self, study, script_history):
        super().__init__()
        # window
        self.child_list = []
        self.minsize(900, 400)
        self.title('Dashboard')
        self.init_menu()
        self.window_exist = True
        # panel
        self.columnconfigure([0, 1, 2], weight=1)
        self.rowconfigure([0, 1], weight=1)

        self.dataset_panel = DatasetPanel(self, row=0, column=0)
        self.preprocess_panel = PreprocessPanel(self, row=0, column=1)
        self.training_scheme_panel = TrainingSchemePanel(self, row=0, column=2)
        self.training_setting_panel = TrainingSettingPanel(
            self, row=1, column=0, columnspan=2
        )
        self.explanation_setting_panel = ExplanationSettingPanel(self, row=2, column=0, columnspan=2)
        self.training_status_panel = TrainingStatusPanel(self, row=1, column=2, rowspan=2)

        self.study = study
        self.script_history = script_history
        self.clear_script()

        self.after(1, self.update_dashboard)
        self.after(1, lambda: self.update_dashboard(loop=True))

    def update_dashboard(self, loop=False):
        if not self.window_exist:
            return
        if not loop:
            self.update_idletasks()
            self.dataset_panel.update_panel(self.study.preprocessed_data_list)
            self.update_idletasks()
            self.preprocess_panel.update_panel(self.study.preprocessed_data_list)
            self.update_idletasks()
            self.training_scheme_panel.update_panel(self.study.datasets)
            self.update_idletasks()
            self.training_setting_panel.update_panel(
                self.study.model_holder, self.study.training_option
            )
            self.explanation_setting_panel.update_panel(self.study.get_saliency_params())
            self.update_idletasks()

        self.training_status_panel.update_panel(self.study.trainer)
        self.update_idletasks()
        if loop:
            self.after(2000, lambda: self.update_dashboard(loop=loop))

    def init_menu(self):
        menu = tk.Menu(self, tearoff=0)
        # Top level
        import_data_menu = tk.Menu(menu, tearoff=0)
        preprocess_menu = tk.Menu(menu, tearoff=0)
        training_menu = tk.Menu(menu, tearoff=0)
        evaluation_menu = tk.Menu(menu, tearoff=0)
        visualization_menu = tk.Menu(menu, tearoff=0)
        script_menu = tk.Menu(menu, tearoff=0)

        menu.add_cascade(label="Import data", menu=import_data_menu)
        menu.add_cascade(label="Preprocess", menu=preprocess_menu)
        menu.add_cascade(label="Training", menu=training_menu)
        menu.add_cascade(label="Evaluation", menu=evaluation_menu)
        menu.add_cascade(label="Visualization", menu=visualization_menu)
        menu.add_cascade(label="Script", menu=script_menu)

        # import data
        for import_module in IMPORT_TYPE_MODULE_LIST:
            import_data_menu.add_command(
                label=import_module.command_label,
                command=lambda var=import_module: self.import_data(var)
            )

        # preprocess/epoching
        edit_event_menu = None
        for preprocess_module in PREPROCESS_MODULE_LIST:
            if "Edit Event" not in preprocess_module.command_label:
                preprocess_menu.add_command(
                    label=preprocess_module.command_label,
                    command=lambda var=preprocess_module: self.preprocess(var)
                )
            else:
                if edit_event_menu is None:
                    edit_event_menu = tk.Menu(preprocess_menu, tearoff=0)
                    preprocess_menu.add_cascade(
                        label='Edit Event', menu=edit_event_menu
                    )
                edit_event_menu.add_command(
                    label=preprocess_module.command_label,
                    command=lambda var=preprocess_module: self.preprocess(var)
                )
        preprocess_menu.add_command(label='Reset', command=self.reset_preprocess)

        # training
        training_setting_menu = tk.Menu(training_menu, tearoff=0)
        training_menu.add_command(label='Dataset Splitting', command=self.split_data)
        training_menu.add_command(label='Model Selection', command=self.select_model)
        training_menu.add_cascade(label='Training Setting', menu=training_setting_menu)
        training_menu.add_command(
            label='Generate Training Plan', command=self.generate_plan
        )
        training_menu.add_command(
            label='Training Manager', command=self.open_training_manager
        )
        # training setting
        training_setting_menu.add_command(
            label='Training', command=self.training_setting
        )
        training_setting_menu.add_command(
            label='Test Only', command=self.test_only_setting
        )

        # evaluation
        for evaluation_module in EVALUATION_MODULE_LIST:
            evaluation_menu.add_command(
                label=evaluation_module.command_label,
                command=lambda var=evaluation_module: self.evaluate(var)
            )

        # visualization
        visualization_menu.add_command(label='Set Montage', command=self.set_montage)
        visualization_menu.add_command(label='Set Saliency Methods', command=self.set_saliency)
        for visualization_module in VISUALIZATION_MODULE_LIST:
            visualization_menu.add_command(
                label=visualization_module.command_label,
                command=lambda var=visualization_module: self.visualize(var)
            )
        visualization_menu.add_command(label='clean plots', command=self.clean_plot)

        # script
        script_menu.add_command(
            label='Show command script',
            command=lambda: self.show_script(ScriptType.CLI)
        )
        script_menu.add_command(
            label='Show ui script',
            command=lambda: self.show_script(ScriptType.UI)
        )
        script_menu.add_command(
            label='Show all script',
            command=lambda: self.show_script(ScriptType.ALL)
        )
        script_menu.add_command(label='Clear script', command=self.clear_script)

        self.config(menu=menu)

    def warn_flow_cleaning(self):
        if tk.messagebox.askokcancel(
            parent=self, title='Warning',
            message=(
                'This step has already been done, '
                'all following data will be removed if you reset this step.\n'
                'Do you want to proceed?'
            )
        ):
            return True
        return False

    # data
    def import_data(self, import_module):
        if self.study.should_clean_raw_data(interact=False):
            if not self.warn_flow_cleaning():
                return
            self.study.clean_raw_data(force_update=True)
            self.script_history.add_cmd('study.clean_raw_data(force_update=True)')
            self.update_dashboard()

        current_import_module = import_module(self)
        data_loader = current_import_module.get_result()
        if data_loader:
            self.script_history += current_import_module.get_script_history()
            data_loader.apply(self.study)
            self.script_history.add_cmd('data_loader.apply(study)')
            self.update_dashboard()

    def preprocess(self, preprocess_module):
        if not self.clean_datasets():
            return

        current_preprocess_module = preprocess_module(
            self, self.study.preprocessed_data_list
        )
        preprocessed_data_list = current_preprocess_module.get_result()
        if preprocessed_data_list:
            self.study.set_preprocessed_data_list(preprocessed_data_list)
            self.script_history += current_preprocess_module.get_script_history()
            self.update_dashboard()

    def reset_preprocess(self):
        if not self.clean_datasets():
            return

        try:
            self.study.reset_preprocess()
            tk.messagebox.showinfo(parent=self, title='Success', message='OK')
            self.script_history.add_cmd('study.reset_preprocess()')
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e
        self.update_dashboard()

    # train
    def split_data(self):
        if not self.clean_datasets():
            return

        data_splitting_module = DataSplittingSettingWindow(self, self.study.epoch_data)
        datasets_generator = data_splitting_module.get_result()
        if datasets_generator:
            datasets_generator.apply(self.study)
            data_splitting_script = data_splitting_module.get_script_history()
            self.script_history += data_splitting_script
            self.script_history.add_cmd(
                'datasets_generator.apply(study)', newline=True
            )
            self.update_dashboard()

    def select_model(self):
        if not self.clean_trainer():
            return

        model_selection_module = ModelSelectionWindow(self)
        model_holder = model_selection_module.get_result()
        if model_holder:
            self.study.set_model_holder(model_holder)

            model_selection_script = model_selection_module.get_script_history()
            self.script_history += model_selection_script
            self.script_history.add_cmd('study.set_model_holder(model_holder)')

            self.update_dashboard()

    def training_setting(self):
        if not self.clean_trainer():
            return

        training_module = TrainingSettingWindow(self)
        training_option = training_module.get_result()
        if training_option:
            self.study.set_training_option(training_option)

            training_module_script = training_module.get_script_history()
            self.script_history += training_module_script
            self.script_history.add_cmd('study.set_training_option(training_option)')

            self.update_dashboard()

    def test_only_setting(self):
        if not self.clean_trainer():
            return

        training_module = TestOnlySettingWindow(self)
        training_option = training_module.get_result()
        if training_option:
            self.study.set_training_option(training_option)

            training_module_script = training_module.get_script_history()
            self.script_history += training_module_script
            self.script_history.add_cmd('study.set_training_option(training_option)')

            self.update_dashboard()

    def generate_plan(self):
        if not self.clean_trainer():
            return

        try:
            self.study.generate_plan()
            self.script_history.add_cmd('study.generate_plan()', newline=True)
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e

        self.update_dashboard()
        self.open_training_manager()

    def open_training_manager(self):
        history = TrainingManagerWindow(self, self.study.trainer).get_script_history()
        self.script_history += history

    # eval
    def evaluate(self, evaluation_module):
        training_plan_holders = None
        if self.study.trainer:
            training_plan_holders = self.study.trainer.get_training_plan_holders()
        history = evaluation_module(self, training_plan_holders).get_script_history()
        self.script_history += history

    # visualize
    def set_montage(self):
        if type(self.study.epoch_data) != Epochs:
            raise ValidateException(
                window=self, message='No valid epoch data is generated'
            )
        pick_montage_module = PickMontageWindow(
            self, self.study.epoch_data.get_channel_names()
        )
        chs, positions = pick_montage_module.get_result()
        if chs is not None and positions is not None:
            self.study.set_channels(chs, positions)

            self.script_history += pick_montage_module.get_script_history()
            self.script_history.add_cmd("study.set_channels(chs, positions)")
            self.update_dashboard()
    
    def set_saliency(self):
        if self.study.trainer:
            if self.study.trainer.get_training_plan_holders()[0].get_plans()[0].is_finished() or self.study.get_saliency_params() is not None:
                if not tk.messagebox.askokcancel(
                    parent=self, title='Warning',
                    message=(
                        'The saliency maps are already computed,\n'
                        'all saliency maps will be recomputed if you reset this step.\n'
                        'Do you want to proceed?'
                    )
                ):
                    return
        set_saliency_module = SetSaliencyWindow(self, self.study.get_saliency_params())
        saliency_param_confirm, saliency_params = set_saliency_module.get_result()
        if saliency_param_confirm:
            self.study.set_saliency_params(saliency_params)
            self.script_history += set_saliency_module.get_script_history()
            self.script_history.add_cmd("study.set_saliency_params(saliency_params)")
            self.update_dashboard()

    def clean_plot(self):
        # could crash system called in threads
        plt.close('all')

    def visualize(self, visualization_module):
        training_plan_holders = None
        if self.study.trainer:
            training_plan_holders = self.study.trainer.get_training_plan_holders()
        hist = visualization_module(self, training_plan_holders).get_script_history()
        self.script_history += hist

    def show_script(self, script_type, target=None):
        if target is None:
            target = self
            ui_script = deepcopy(target.script_history)
        else:
            ui_script = Script()

        for child in target.child_list:
            if child.window_exist:
                ui_script += child._get_script_history()
                ui_script += self.show_script(script_type, target=child)

        if target == self:
            ScriptPreview(self, ui_script, script_type)
        else:
            return ui_script

    def clear_script(self):
        if not self.script_history:
            self.script_history = Script()
        self.script_history.reset()
        self.script_history.add_import('from XBrainLab import Study')
        self.script_history.add_import('from XBrainLab.ui import XBrainLab')

        if self.study.loaded_data_list:
            self.script_history.add_cmd('# study = Study()')
        else:
            self.script_history.add_cmd('study = Study()')

        self.script_history.add_ui_cmd('study = Study()')
        self.script_history.add_ui_cmd('lab = XBrainLab(study)')

    def get_script(self):
        return self.script_history

    # clean
    def clean_datasets(self):
        if self.study.should_clean_datasets(interact=False):
            if not self.warn_flow_cleaning():
                return False
            self.study.clean_datasets(force_update=True)
            self.script_history.add_cmd('study.clean_datasets(force_update=True)')
            self.update_dashboard()
        return True

    def clean_trainer(self):
        if self.study.should_clean_trainer(interact=False):
            if not self.warn_flow_cleaning():
                return False
            self.study.clean_trainer(force_update=True)
            self.script_history.add_cmd('study.clean_trainer(force_update=True)')
            self.update_dashboard()
        return True

    # destroy
    def append_child_window(self, child):
        self.child_list.append(child)

    def remove_child_window(self, child):
        self.child_list.remove(child)

    def check_training(self, force=False):
        if not self.study.is_training():
            return
        if (
            tk.messagebox.askyesno(
                parent=self, title='Warning',
                message='Training is in progress.\nDo you want to interrupt it?'
            ) or
            force
        ):
            self.study.stop_training()

    def destroy(self, force=False):
        self.check_training(force)

        child_list = self.child_list.copy()
        for child in child_list:
            if not child.destroy(force):
                return False

        plt.close('all')
        # recycling
        print('recycling...')
        self.withdraw()
        self.window_exist = False
        self.after(3000, super().destroy)
