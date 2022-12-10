import numpy as np
import time
import threading

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox

from ..base import TopWindow, ValidateException
from ..script import Script

from XBrainLab.dataset import TrainingType, SplitByType, ValSplitByType, SplitUnit
from XBrainLab.dataset import DataSplitter, DataSplittingConfig, DatasetGenerator
from XBrainLab.dataset import Epochs, Dataset

DEFAULT_SPLIT_ENTRY_VALUE = 0.2
LOADING_TREE_ROW_IID = '-'
###
class DataSplittingWindow(TopWindow):
    def __init__(self, parent, title, epoch_data, config):
        super().__init__(parent, title)
        self.epoch_data = epoch_data
        self.config = config
        self.check_data()
        self.datasets = []
        self.return_datasets = None
        self.script_history = Script()
        self.ret_script_history = None
        #
        self.dataset_generator = None
        self.preview_worker = None
        self.preview_failed = False
        self.last_update = time.time()
        #
        split_unit_list = [i.value for i in SplitUnit if i != SplitUnit.KFOLD]
        # preprocess
        val_splitter_list, test_splitter_list = self.config.generate_splitter_option()
        # treeview
        columns = ['select', 'name', 'train', 'val', 'test']
        tree_frame = tk.Frame(self)
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode=tk.BROWSE)
        for i in columns:
            tree.heading(i, text=i)
            tree.column(i, width=80, anchor=tk.CENTER)

        tree.pack(fill=tk.BOTH, expand=True)
        tk.Button(tree_frame, text='Show info', command=self.show_info).pack(padx=20)
        
        # dataset frame
        dataset_frame = tk.LabelFrame(self, text ='Dataset Info')
        tk.Label(dataset_frame, text='Subject: ')                        .grid(row=0, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(epoch_data.subject_map)).grid(row=0, column=1, padx=5)
        tk.Label(dataset_frame, text='Session: ')                        .grid(row=1, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(epoch_data.session_map)).grid(row=1, column=1, padx=5)
        tk.Label(dataset_frame, text='Label: ')                          .grid(row=2, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(epoch_data.label_map))  .grid(row=2, column=1, padx=5)
        tk.Label(dataset_frame, text='Trial: ')                          .grid(row=3, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(epoch_data.data))              .grid(row=3, column=1, padx=5)
        # training
        training_frame = tk.LabelFrame(self, text ='Training type')
        tk.Label(training_frame, text=self.config.train_type.value).pack(pady=5)

        # val frame
        validation_frame = tk.LabelFrame(self, text ='Validation')
        validation_frame.grid_columnconfigure(0, weight=1)
        row = 0
        for val_splitter in val_splitter_list:
            if val_splitter.is_option:
                ## init variables
                val_splitter.set_split_unit_var(var=tk.StringVar(self), val=split_unit_list[0], callback=self.preview)
                val_splitter.set_entry_var(var=tk.StringVar(self), val=DEFAULT_SPLIT_ENTRY_VALUE, callback=self.preview)
                ## init widget
                val_split_by_label = tk.Label(validation_frame, text=val_splitter.text)
                val_split_type_option = tk.OptionMenu(validation_frame, val_splitter.split_var, *split_unit_list)
                val_split_entry = tk.Entry(validation_frame, textvariable=val_splitter.entry_var)
                ## pack
                val_split_by_label.grid(row=row + 0, column=0, columnspan=2)
                val_split_type_option.grid(row=row + 1, column=0)
                val_split_entry.grid(row=row + 1, column=1)
                row += 2
            else:
                tk.Label(validation_frame, text=val_splitter.text).grid(row=row, column=0, columnspan=2, pady=5)

        # test frame
        testing_frame = tk.LabelFrame(self, text='Testing')
        testing_frame.grid_columnconfigure(0, weight=1)
        row = 0
        if config.is_cross_validation:
            tk.Label(testing_frame, text='Cross Validation').grid(row=row, column=0, columnspan=2)
            row += 1
        idx = 0
        for test_splitter in test_splitter_list:
            if test_splitter.is_option:
                idx += 1
                ## init variables
                test_splitter.set_split_unit_var(var=tk.StringVar(self), val=split_unit_list[0], callback=self.preview)
                test_splitter.set_entry_var(var=tk.StringVar(self), val=DEFAULT_SPLIT_ENTRY_VALUE, callback=self.preview)
                ## init widget
                test_split_by_label = tk.Label(testing_frame, text=test_splitter.text)
                tmp_split_unit_list = split_unit_list
                if (config.is_cross_validation and idx == 1):
                    tmp_split_unit_list = split_unit_list + [SplitUnit.KFOLD.value]
                test_split_type_option = tk.OptionMenu(testing_frame, test_splitter.split_var, *tmp_split_unit_list)
                test_split_entry = tk.Entry(testing_frame, textvariable=test_splitter.entry_var)
                ## pack
                test_split_by_label.grid(row=row + 0, column=0, columnspan=2)
                test_split_type_option.grid(row=row + 1, column=0)
                test_split_entry.grid(row=row + 1, column=1)
                row += 2
            else:
                tk.Label(testing_frame, text=test_splitter.text).grid(row=row, column=0, columnspan=2, pady=5)
                row += 1

        # btn
        confirm_btn = tk.Button(self, text='Confirm', command=self.confirm)

        # pack
        tree_frame.pack(side=tk.LEFT, padx=20, anchor='n', pady=40, fill=tk.BOTH, expand=True)
        dataset_frame.pack(side=tk.TOP, padx=20, pady=20, fill=tk.X, expand=True)
        training_frame.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)
        testing_frame.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)
        validation_frame.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)
        confirm_btn.pack(side=tk.TOP, padx=20, pady=20)

        # init 
        self.val_splitter_list = val_splitter_list
        self.test_splitter_list = test_splitter_list
        self.tree = tree
        
        ## init func
        self.preview()
        self.update_table()

    def check_data(self):
        if type(self.epoch_data) != Epochs:
            raise InitWindowValidateException(self, 'No valid epoch data is generated')
        if type(self.config) != DataSplittingConfigHolder:
            raise InitWindowValidateException(self, 'No valid data splitting config is generated')

    def preview(self, var=None, index=None, mode=None):
        # reset config
        self.datasets = []
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", 'end', iid=LOADING_TREE_ROW_IID, values=['...'] + ['calculating'] + ['...'] * 3)
        # convert variable to constant
        for splitter in self.test_splitter_list:
            splitter.to_thread()
        for splitter in self.val_splitter_list:
            splitter.to_thread()        
        # start worker
        if self.dataset_generator:
            self.dataset_generator.set_interrupt()
        
        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.dataset import DataSplitter, DataSplittingConfig")
        self.script_history.add_import("from XBrainLab.dataset import SplitUnit, TrainingType, SplitByType, ValSplitByType")

        self.script_history.add_cmd("test_splitter_list = [")
        for test_splitter in self.test_splitter_list:
            self.script_history.add_cmd(f"DataSplitter(split_type={test_splitter.get_split_type_repr()}, value_var={repr(test_splitter.get_raw_value())}, split_unit={test_splitter.get_split_unit_repr()}),")
        self.script_history.add_cmd("]")

        self.script_history.add_cmd("val_splitter_list = [")
        for val_splitter in self.val_splitter_list:
            self.script_history.add_cmd(f"DataSplitter(split_type={val_splitter.get_split_type_repr()}, value_var={repr(val_splitter.get_raw_value())}, split_unit={val_splitter.get_split_unit_repr()}),")
        self.script_history.add_cmd("]")

        self.script_history.add_cmd(f"datasets_config = DataSplittingConfig(train_type={self.config.get_train_type_repr()}, is_cross_validation={repr(self.config.is_cross_validation)}, " + 
        f"val_splitter_list=val_splitter_list, test_splitter_list=test_splitter_list)")

        self.script_history.add_cmd("datasets_generator = study.get_datasets_generator(config=datasets_config)")

        self.dataset_generator = DatasetGenerator(self.epoch_data, config=self.config, datasets=self.datasets)
        self.preview_worker = threading.Thread(target=self.dataset_generator.generate)
        self.preview_worker.start()
    
    def is_preview_failed(self):
        if self.dataset_generator:
            return self.dataset_generator.preview_failed
        return False
    
    def _failed_preview(self):
        self.datasets = []
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", 'end', iid='Nan', values=['Nan'] * 5)

    def update_table(self):
        if self.is_preview_failed():
            self._failed_preview()
        elif len(self.datasets) > 0:
            if LOADING_TREE_ROW_IID in self.tree.get_children():
                self.tree.delete(LOADING_TREE_ROW_IID)
            counter = 0
            while len(self.tree.get_children()) < len(self.datasets):
                if counter > 50:
                    break
                counter += 1
                idx = len(self.tree.get_children())
                dataset = self.datasets[idx]
                self.tree.insert("", 'end', iid=idx, values=dataset.get_treeview_row_info())
        self.after(500, self.update_table)

    def show_info(self):
        if len(self.datasets) == 0 or self.tree.focus() == '' or not self.tree.focus().isdigit():
            raise ValidateException(window=self, message='No valid item is selected')
        idx = int(self.tree.focus())
        target = self.datasets[idx]
        window = DataSplittingInfoWindow(self, target)
        show_info_script = window._get_script_history()
        if show_info_script:
            self.script_history.add_cmd(f"dataset = dataset[{repr(idx)}]")
            self.script_history += show_info_script
        
        # update tree
        if len(self.datasets) > idx:
            target = self.datasets[idx]
            if self.window_exist:
                self.tree.item(idx, values=target.get_treeview_row_info())

    def confirm(self):
        # check if data is empty
        for dataset in self.datasets:
            if dataset.has_set_empty():
                if tk.messagebox.askokcancel(parent=self, title='Warning', message='There are some datasets without training/testing/validation data.\nDo you want to proceed?'):
                    break
                else:
                    return
        if self.preview_worker.is_alive():
            tk.messagebox.showinfo(parent=self, title='Warning', message='Generating dataset, please try again later.')
            return
        # remove unselected plan
        try:
            self.dataset_generator.prepare_reuslt()
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        
        self.return_datasets = self.dataset_generator
        self.ret_script_history = self.script_history
        self.destroy()

    def _get_result(self):
        return self.return_datasets
    
    def _get_script_history(self):
        return self.ret_script_history

    def destroy(self, force=False):
        if self.dataset_generator:
            self.dataset_generator.set_interrupt()
        return super().destroy(force)
##
class DataSplittingInfoWindow(TopWindow):
    def __init__(self, parent, dataset):
        super().__init__(parent, 'Data splitting Info')
        self.dataset = dataset
        self.check_data()
        self.script_history = Script()
        self.ret_script_history = None
        
        name_var = tk.StringVar(self)
        select_var = tk.BooleanVar(self)
        # init value
        name_var.set(dataset.name)
        select_var.set(dataset.is_selected)

        select_checkbox = tk.Checkbutton(self, text='select', var=select_var)

        name_label = tk.Label(self, text='Name')
        name_entry = tk.Entry(self, textvariable=name_var)

        train_frame = tk.LabelFrame(self, text='Train')
        val_frame = tk.LabelFrame(self, text='Val')
        test_frame = tk.LabelFrame(self, text='Test')

        train_tree = ttk.Treeview(train_frame)
        val_tree = ttk.Treeview(val_frame)
        test_tree = ttk.Treeview(test_frame)
        
        train_tree.pack(fill=tk.BOTH, expand=True)
        val_tree.pack(fill=tk.BOTH, expand=True)
        test_tree.pack(fill=tk.BOTH, expand=True)

        select_checkbox.grid(row=0, column=1)
        name_label.grid(row=1, column=0, sticky='e')
        name_entry.grid(row=1, column=1, sticky='we')
        train_frame.grid(row=2, column=0, sticky='news')
        val_frame.grid(row=2, column=1, sticky='news')
        test_frame.grid(row=2, column=2, sticky='news')
        self.columnconfigure([0,1,2], weight=1)
        self.rowconfigure([2], weight=1)


        tk.Button(self, text='Save', command=self.confirm).grid(row=3, column=1)
        self.name_var = name_var
        self.select_var = select_var
        self.show_info(train_tree, dataset.train_mask)
        self.show_info(test_tree, dataset.test_mask)
        self.show_info(val_tree, dataset.val_mask)

    def check_data(self):
        if type(self.dataset) != Dataset:
            raise InitWindowValidateException(self, 'No a valid dataset')
        
    def show_info(self, tree, mask):
        epoch_data = self.dataset.get_epoch_data()
        # traverse subject
        for subject_idx in np.unique( epoch_data.get_subject_list_by_mask(mask) ):
            subject_mask = (epoch_data.get_subject_list() == subject_idx) & mask
            subject_root = tree.insert("", 'end', text=f"Subject {epoch_data.get_subject_name(subject_idx)} ({sum(subject_mask)})")
            # traverse session
            for session_idx in np.unique( epoch_data.get_session_list_by_mask(subject_mask) ):
                session_mask = (epoch_data.get_session_list() == session_idx) & subject_mask
                session_root = tree.insert(subject_root, 'end', text=f"Session {epoch_data.get_session_name(session_idx)} ({sum(session_mask)})")
                # traverse label
                for label_idx in np.unique( epoch_data.get_label_list_by_mask(session_mask) ):
                    label_mask = (epoch_data.get_label_list() == label_idx) & session_mask
                    label_root = tree.insert(session_root, 'end', text=f"Label {epoch_data.get_label_name(label_idx)} ({sum(label_mask)})")
                    # traverse index
                    idx_list = epoch_data.get_idx_list_by_mask(label_mask)
                    start_idx = 0
                    last_idx = None
                    for i in idx_list:
                        if last_idx is None:
                            last_idx = i
                            start_idx = i
                            continue
                        if last_idx + 1 != i:
                            tree.insert(label_root, 'end', text=f"Trial {int(start_idx)}~{int(last_idx)}")
                            start_idx = i
                        last_idx = i
                    if last_idx:
                        tree.insert(label_root, 'end', text=f"Trial {int(start_idx)}~{int(idx_list[-1])}")
    
    def confirm(self):
        self.script_history = Script()
        if self.dataset.get_ori_name() != self.name_var.get():
            self.dataset.set_name(self.name_var.get())
            self.script_history.add_cmd(f"dataset.set_name({repr(self.name_var.get())})")

        
        if self.dataset.is_selected != self.select_var.get():
            self.dataset.set_selection(self.select_var.get())
            self.script_history.add_cmd(f"dataset.set_selection({repr(self.select_var.get())})")

        
        self.ret_script_history = self.script_history
        self.destroy()
    
    def _get_script_history(self):
        return self.ret_script_history


###

class DataSplitterHolder(DataSplitter):
    def __init__(self, is_option, split_type):
        super().__init__(split_type, value_var=None, split_unit=None, is_option=is_option)
        self.split_var = None
        self.entry_var = None

        self.is_valid_var = False
    
    # initialize variable
    def set_split_unit_var(self, var, val, callback):
        self.split_var = var
        self.split_var.set(val)
        self.split_var.trace_add('write', callback)

    def set_entry_var(self, var, val, callback):
        self.entry_var = var
        self.entry_var.set(val)
        self.entry_var.trace_add('write', callback)

    # convert variable to constant
    def _is_valid(self):
        if self.entry_var is None:
            return False
        if self.split_var is None:
            return False
        if self.split_var.get() == SplitUnit.RATIO.value:
            try:
                val = float(self.entry_var.get())
                if 0 <= val <= 1:
                    return True
            except ValueError:
                return False   
        elif self.split_var.get() == SplitUnit.NUMBER.value:
            return self.entry_var.get().isdigit()
        elif self.split_var.get() == SplitUnit.KFOLD.value:
            val = self.entry_var.get()
            if val.isdigit():
                return 0 < int(val)
        return False

    def _get_value(self):
        if not self._is_valid():
            return 0
        return self.entry_var.get()

    def _get_split_unit(self):
        if self.split_var is None:
            return None
        for i in SplitUnit:
            if i.value == self.split_var.get():
                return i
        
    def to_thread(self):
        self.is_valid_var = self._is_valid()
        self.value_var = self._get_value()
        self.split_unit = self._get_split_unit()
    #
    def is_valid(self):
        return self.is_valid_var

class DataSplittingConfigHolder(DataSplittingConfig):
    def __init__(self, train_type, val_type_list, test_type_list, is_cross_validation):
        super().__init__(train_type, is_cross_validation, val_splitter_list=None, test_splitter_list=None)
        self.train_type = train_type # TrainingType
        self.val_type_list = val_type_list # [SplitByType ...]
        self.test_type_list = test_type_list # [ValSplitByType ...]
    
    def generate_splitter_option(self):
        if not self.val_splitter_list:
            val_splitter_list = []
            for val_type in self.val_type_list:
                is_option = not (val_type == ValSplitByType.DISABLE)
                val_splitter_list.append(DataSplitterHolder(is_option, val_type))
            test_splitter_list = []
            for test_type in self.test_type_list:
                is_option = not (test_type == SplitByType.DISABLE)
                test_splitter_list.append(DataSplitterHolder(is_option, test_type))
            self.val_splitter_list = val_splitter_list
            self.test_splitter_list = test_splitter_list
        return self.val_splitter_list, self.test_splitter_list
