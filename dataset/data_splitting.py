import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from ..base import TopWindow, ValidateException
from .data_holder import DataSet
from .option import TrainingType, SplitByType, ValSplitByType, SplitUnit
import numpy as np
import time
import threading

DEFAULT_SPLIT_ENTRY_VALUE = 0.2
LOADING_TREE_ROW_IID = '-'
###
class DataSplittingWindow(TopWindow):
    def __init__(self, parent, title, data_holder, config):
        super().__init__(parent, title)
        self.data_holder = data_holder
        self.config = config
        self.datasets = []
        self.return_datasets = None
        #
        self.preview_worker = None
        self.preview_failed = False
        self.last_update = time.time()
        #
        split_unit_list = [i.value for i in SplitUnit if i != SplitUnit.KFOLD]
        # preprocess
        val_splitter_list, test_splitter_list = self.config.get_splitter_option()
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
        tk.Label(dataset_frame, text=len(data_holder.subject_map)).grid(row=0, column=1, padx=5)
        tk.Label(dataset_frame, text='Session: ')                        .grid(row=1, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.session_map)).grid(row=1, column=1, padx=5)
        tk.Label(dataset_frame, text='Label: ')                          .grid(row=2, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.label_map))  .grid(row=2, column=1, padx=5)
        tk.Label(dataset_frame, text='Trail: ')                          .grid(row=3, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.data))              .grid(row=3, column=1, padx=5)
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
                val_splitter.set_split_unit_var(root=self, val=split_unit_list[0], callback=self.preview)
                val_splitter.set_entry_var(root=self, val=DEFAULT_SPLIT_ENTRY_VALUE, callback=self.preview)
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
                test_splitter.set_split_unit_var(root=self, val=split_unit_list[0], callback=self.preview)
                test_splitter.set_entry_var(root=self, val=DEFAULT_SPLIT_ENTRY_VALUE, callback=self.preview)
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

    def preview(self, var=None, index=None, mode=None):
        # reset config
        self.preview_failed = False
        self.last_update = time.time()
        self.datasets = []
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", 'end', iid=LOADING_TREE_ROW_IID, values=['...'] + ['calculating'] + ['...'] * 3)
        # convert variable to constant
        for splitter in self.test_splitter_list:
            splitter.to_thread()
        for splitter in self.val_splitter_list:
            splitter.to_thread()        
        # start worker
        self.preview_worker = threading.Thread(target=self.handle_data, args=(self.last_update, ))
        self.preview_worker.start()
    
    # TODO code review
    def handle_data(self, checker):
        # for loop for individual scheme
        # break at the end if not individual scheme
        for subject_idx in range(len(self.data_holder.get_subject_index_list())):
            group_idx = 0
            # parms for cross validation
            # break at the end if not cross validation
            has_next = True
            ref_mask = None
            ref_exclude = None
            while has_next:
                # check job interrupt
                if self.last_update != checker:
                    return
                dataset = DataSet(self.data_holder, self.config)
                dataset.set_name(f"Group {group_idx}")
                # set name to subject-xxx for individual scheme
                if self.config.train_type == TrainingType.IND:
                    dataset.set_remaining_by_subject_idx(subject_idx)
                    if group_idx > 0:
                        dataset.set_name(f"Subject {self.data_holder.get_subject_name(subject_idx)}-{group_idx}")
                    else:
                        dataset.set_name(f"Subject {self.data_holder.get_subject_name(subject_idx)}")
                # get reference mask
                if ref_mask is None:
                    mask = dataset.get_remaining_mask()
                    ref_exclude = np.logical_not(mask)
                else:
                    mask = ref_mask
                    ref_exclude = dataset.get_remaining_mask() & np.logical_not(ref_mask)
                # filter out non-target subjects for individual scheme
                if self.config.train_type == TrainingType.IND:
                    ref_exclude = dataset.intersection_with_subject_by_idx(ref_exclude, subject_idx)
                # split for test
                idx = 0
                for test_splitter in self.test_splitter_list:
                    if test_splitter.is_option:
                        if self.last_update != checker:
                            return
                        if not test_splitter.is_valid():
                            self.preview_failed = True
                            return
                        # session
                        if test_splitter.split_type == SplitByType.SESSION or test_splitter.split_type == SplitByType.SESSION_IND:
                            mask, exclude = self.data_holder.pick_session(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.SESSION_IND:
                                dataset.discard(exclude)
                            elif idx == 0:
                                # filter out first option from validation data
                                dataset.kept_training_session(mask)
                        # label
                        elif test_splitter.split_type == SplitByType.TRAIL or test_splitter.split_type == SplitByType.TRAIL_IND:
                            mask, exclude = self.data_holder.pick_trail(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.TRAIL_IND:
                                dataset.discard(exclude)
                        # subject
                        elif test_splitter.split_type == SplitByType.SUBJECT or test_splitter.split_type == SplitByType.SUBJECT_IND:
                            mask, exclude = self.data_holder.pick_subject(mask, num=test_splitter.get_value(), group_idx=group_idx, split_unit=test_splitter.get_split_unit(), ref_exclude=ref_exclude if (idx == 0) else None)
                            # save for next cross validation
                            if idx == 0:
                                ref_mask = exclude.copy()
                                # restore previous cross validation part
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.split_type == SplitByType.SUBJECT_IND:
                                dataset.discard(exclude)
                            elif idx == 0:
                                dataset.kept_training_subject(mask)
                        idx += 1

                # set result as mask if available
                if not has_next:
                    break
                if idx > 0:
                    dataset.set_test(mask)
                
                # split val data
                mask = dataset.get_remaining_mask()
                idx = 0
                for val_splitter in self.val_splitter_list:
                    if val_splitter.is_option:
                        # check job interrupt
                        if self.last_update != checker:
                            return
                        if not val_splitter.is_valid():
                            self.preview_failed = True
                            return
                        # session
                        if val_splitter.split_type == ValSplitByType.SESSION:
                            mask, exclude = self.data_holder.pick_session(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        # label
                        elif val_splitter.split_type == ValSplitByType.TRAIL:
                            mask, exclude = self.data_holder.pick_trail(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        # subject
                        elif val_splitter.split_type == ValSplitByType.SUBJECT:
                            mask, exclude = self.data_holder.pick_subject(mask, num=val_splitter.get_value(), split_unit=val_splitter.get_split_unit())
                        idx += 1
                
                # set result as mask if available
                if idx > 0:
                    dataset.set_val(mask)
                dataset.set_train()
                # check job interrupt
                if self.last_update != checker:
                    return
                self.datasets.append(dataset)
                group_idx += 1
                # break at the end if not cross validation
                if not self.config.is_cross_validation:
                    break
            # break at the end if not individual scheme
            if self.config.train_type != TrainingType.IND:
                break
        
        if len(self.datasets) == 0:
            self.preview_failed = True

    def _failed_preview(self):
        self.datasets = []
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", 'end', iid='Nan', values=['Nan'] * 5)

    def update_table(self):
        if self.preview_failed:
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
        window.get_result()
        # update tree
        if len(self.datasets) > idx:
            target = self.datasets[idx]
            if self.winfo_exists():
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
        while True:
            done = True
            for i in range(len(self.datasets)):
                if not self.datasets[i].is_selected:
                    del self.datasets[i]
                    done = False
                    break
            if done:
                break
        # check if dataset is empty
        if len(self.datasets) == 0:
            raise ValidateException(window=self, message='No valid dataset is generated')

        self.return_datasets = self.datasets
        self.destroy()

    def _get_result(self):
        return self.return_datasets
    
    def destroy(self):
        self.last_update = time.time() + 1000
        super().destroy()
##
class DataSplittingInfoWindow(TopWindow):
    def __init__(self, parent, dataset):
        super().__init__(parent, 'Data splitting Info')
        self.dataset = dataset
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

    def show_info(self, tree, mask):
        data_holder = self.dataset.get_data_holder()
        # traverse subject
        for subject_idx in np.unique( data_holder.get_subject_list_by_mask(mask) ):
            subject_mask = (data_holder.get_subject_list() == subject_idx) & mask
            subject_root = tree.insert("", 'end', text=f"Subject {data_holder.get_subject_name(subject_idx)} ({sum(subject_mask)})")
            # traverse session
            for session_idx in np.unique( data_holder.get_session_list_by_mask(subject_mask) ):
                session_mask = (data_holder.get_session_list() == session_idx) & subject_mask
                session_root = tree.insert(subject_root, 'end', text=f"Session {data_holder.get_session_name(session_idx)} ({sum(session_mask)})")
                # traverse label
                for label_idx in np.unique( data_holder.get_label_list_by_mask(session_mask) ):
                    label_mask = (data_holder.get_label_list() == label_idx) & session_mask
                    label_root = tree.insert(session_root, 'end', text=f"Label {data_holder.get_label_name(label_idx)} ({sum(label_mask)})")
                    # traverse index
                    idx_list = data_holder.get_idx_list_by_mask(label_mask)
                    start_idx = 0
                    last_idx = None
                    for i in idx_list:
                        if last_idx is None:
                            last_idx = i
                            start_idx = i
                            continue
                        if last_idx + 1 != i:
                            tree.insert(label_root, 'end', text=f"Trail {start_idx}~{last_idx}")
                            start_idx = i
                        last_idx = i
                    if last_idx:
                        tree.insert(label_root, 'end', text=f"Trail {start_idx}~{idx_list[-1]}")
    
    def confirm(self):
        self.dataset.set_name(self.name_var.get())
        self.dataset.set_selection(self.select_var.get())
        self.destroy()

###

class DataSplitter():
    def __init__(self, is_option, text, split_type):
        self.is_option = is_option
        self.split_type = split_type
        self.text = text
        self.split_var = None
        self.entry_var = None

        self.is_valid_var = False
        self.value_var = None
        self.split_unit = None
    
    # initialize variable
    def set_split_unit_var(self, root, val, callback):
        self.split_var = tk.StringVar(root)
        self.split_var.set(val)
        self.split_var.trace_add('write', callback)

    def set_entry_var(self, root, val, callback):
        self.entry_var = tk.StringVar(root)
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
        return float(self.entry_var.get())

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
    # getter
    def get_value(self):
        return self.value_var
        
    def get_split_unit(self):
        return self.split_unit
    
class DataSplittingConfig():
    def __init__(self, train_type, val_type_list, test_type_list, is_cross_validation):
        self.train_type = train_type # TrainingType
        self.val_type_list = val_type_list # [SplitByType ...]
        self.test_type_list = test_type_list # [ValSplitByType ...]
        self.is_cross_validation = is_cross_validation
        self.val_splitter_list = None
        self.test_splitter_list = None
    
    def get_splitter_option(self):
        if not self.val_splitter_list:
            val_splitter_list = []
            for val_type in self.val_type_list:
                is_option = not (val_type == ValSplitByType.DISABLE)
                text = val_type.value
                val_splitter_list.append(DataSplitter(is_option, text, val_type))
            test_splitter_list = []
            for test_type in self.test_type_list:
                is_option = not (test_type == SplitByType.DISABLE)
                text = test_type.value
                test_splitter_list.append(DataSplitter(is_option, text, test_type))
            self.val_splitter_list = val_splitter_list
            self.test_splitter_list = test_splitter_list
        return self.val_splitter_list, self.test_splitter_list

###

if __name__ == '__main__':
    import numpy as np

    from data_holder import Epochs
    from .data_splitting_setting import DataSplittingConfig
    data_holder = Epochs()
    # BCI 4 2a
    data_holder.label = np.array([l for i in range(9) for k in range(2) for l in range(4) for j in range(72)])
    data_holder.session = np.array([k for i in range(9) for k in range(2) for j in range(288)])
    data_holder.subject = np.array([i for i in range(9) for k in range(2) for j in range(288)])
    data_holder.idx = np.array([j for i in range(9) for k in range(2) for j in range(288)])
    data_holder.data = np.random.rand(288*2*9, 22, 1)

    # dementia
    # data_holder.label = np.array([l for l in range(3) for i in range(30) for k in range(2) for j in range(24)])
    # data_holder.session = np.array([k for l in range(3) for i in range(30) for k in range(2) for j in range(24)])
    # data_holder.subject = np.array([l * 30 + i for l in range(3) for i in range(30) for k in range(2) for j in range(24)])
    # data_holder.idx = np.array([j for l in range(3) for i in range(30) for k in range(2) for j in range(24)])
    # data_holder.data = np.random.rand(24*30*2*3, 22, 1)

    ## 
    data_holder.label_map = {i:i for i in np.unique(data_holder.label)}
    data_holder.session_map = {i:i for i in np.unique(data_holder.session)}
    data_holder.subject_map = {i:f"S{i+1}" for i in np.unique(data_holder.subject)}
    train = TrainingType.IND
    val = [ValSplitByType.TRAIL]
    test = [SplitByType.TRAIL]
    config = DataSplittingConfig(train, val, test, True)

    root = tk.Tk()
    root.withdraw()
    window = DataSplittingWindow(root, 'Data splitting', data_holder, config)

    print (window.get_result())
    root.destroy()