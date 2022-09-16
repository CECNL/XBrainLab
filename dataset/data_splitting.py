import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from ..base import TopWindow
from .data_holder import DataSet
from .option import TrainingType, SplitByType, ValSplitByType, SplitUnit
import numpy as np
import time
import threading

###
class DataSplittingWindow(TopWindow):
    def __init__(self, parent, title, data_holder, config):
        super().__init__(parent, title)
        self.last_update = time.time()
        self.data_holder = data_holder
        self.preview_failed = False
        self.config = config
        self.worker = None
        self.datasets = []
        self.return_datasets = None
        split_unit_list = [i.value for i in SplitUnit if i != SplitUnit.KFOLD]
        # preprocess
        val_splitter_list, test_splitter_list = self.config.get_splitter_option()
        # treeview
        columns = ['select', 'name', 'train', 'val', 'test']
        tree_frame = tk.Frame(self)
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode=tk.BROWSE)
        for i in columns:
            tree.heading(i, text=i)
            tree.column(i, width=80)

        tree.pack()
        tk.Button(tree_frame, text='Show info', command=self.show_info).pack()
        
        # dataset frame
        dataset_frame = tk.LabelFrame(self, text ='Dataset Info')
        tk.Label(dataset_frame, text='Subject: ').grid(row=0, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.subject_map.keys())).grid(row=0, column=1, padx=5)
        tk.Label(dataset_frame, text='Session: ').grid(row=1, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.session_map.keys())).grid(row=1, column=1, padx=5)
        tk.Label(dataset_frame, text='Label: ').grid(row=2, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.label_map.keys())).grid(row=2, column=1, padx=5)
        tk.Label(dataset_frame, text='Trail: ').grid(row=3, column=0, sticky='e', padx=3)
        tk.Label(dataset_frame, text=len(data_holder.data)).grid(row=3, column=1, padx=5)
        # training
        training_frame = tk.LabelFrame(self, text ='Training type')
        tk.Label(training_frame, text=self.config.train.value).pack(pady=5)

        # val frame
        validation_frame = tk.LabelFrame(self, text ='Validation')
        validation_frame.grid_columnconfigure(0, weight=1)
        row = 0
        for val_splitter in val_splitter_list:
            if val_splitter.is_option:
                val_splitter.set_split_var(self, split_unit_list[0], self.preview)
                val_split_by_label = tk.Label(validation_frame, text=val_splitter.text)
                val_split_type_option = tk.OptionMenu(validation_frame, val_splitter.split_var, *split_unit_list)
                val_splitter.set_entry_var(self, '0.2', self.preview)
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
                test_splitter.set_split_var(self, split_unit_list[0], self.preview)
                test_split_by_label = tk.Label(testing_frame, text=test_splitter.text)
                tmp_split_unit_list = split_unit_list
                if (config.is_cross_validation and idx == 1):
                    tmp_split_unit_list = split_unit_list + [SplitUnit.KFOLD.value]
                test_split_type_option = tk.OptionMenu(testing_frame, test_splitter.split_var, *tmp_split_unit_list)
                test_splitter.set_entry_var(self, '0.2', self.preview)
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
        tree_frame.pack(side=tk.LEFT, padx=20, anchor='n', pady=40)
        dataset_frame.pack(side=tk.TOP, padx=20, pady=20, fill=tk.X)
        training_frame.pack(side=tk.TOP, padx=20, fill=tk.X)
        testing_frame.pack(side=tk.TOP, padx=20, fill=tk.X)
        validation_frame.pack(side=tk.TOP, padx=20, fill=tk.X)
        confirm_btn.pack(side=tk.TOP, padx=20, pady=20)

        # init 
        self.val_splitter_list = val_splitter_list
        self.test_splitter_list = test_splitter_list
        self.tree = tree
        
        ## init func
        self.preview()
        self.update_table()

    def preview(self, var=None, index=None, mode=None):
        self.preview_failed = False
        self.last_update = time.time()
        self.datasets = []
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", 'end', iid='-', values=['...'] + ['calculating'] + ['...'] * 3)
        for splitter in self.test_splitter_list:
            splitter.to_thread()
        for splitter in self.val_splitter_list:
            splitter.to_thread()        
        self.worker = threading.Thread(target=self.handle_data, args=(self.last_update, ))
        self.worker.start()

    def handle_data(self, checker):
        for subject_idx in range(len(self.data_holder.subject_map.keys())):
            group_idx = 0
            has_next = True
            ref_mask = None
            ref_exclude = None
            while has_next:
                if self.last_update != checker:
                    return
                dataset = DataSet(self.data_holder)
                dataset.set_name(f"Group {group_idx}")
                if self.config.train == TrainingType.IND:
                    dataset.pick_subject_by_idx(subject_idx)
                    if group_idx > 0:
                        dataset.set_name(f"Subject {self.data_holder.subject_map[subject_idx]}-{group_idx}")
                    else:
                        dataset.set_name(f"Subject {self.data_holder.subject_map[subject_idx]}")
                # split test data
                if ref_mask is None:
                    mask = dataset.get_remaining()
                    ref_exclude = np.logical_not(mask)
                else:
                    mask = ref_mask
                    ref_exclude = dataset.get_remaining() & np.logical_not(ref_mask)
                if self.config.train == TrainingType.IND:
                    ref_exclude = dataset.filter_by_subject_idx(ref_exclude, subject_idx)
                idx = 0
                for test_splitter in self.test_splitter_list:
                    if test_splitter.is_option:
                        if self.last_update != checker:
                            return
                        if not test_splitter.is_valid():
                            self.preview_failed = True
                            return
                        # session
                        if test_splitter.option == SplitByType.SESSION or test_splitter.option == SplitByType.SESSION_IND:
                            mask, exclude = dataset.pick_session(mask, num=test_splitter.get_value(), group_idx=group_idx, split_type=test_splitter.get_split_type(), ref_exclude=ref_exclude if (idx == 0) else None)
                            if idx == 0:
                                ref_mask = exclude.copy()
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.option == SplitByType.SESSION_IND:
                                dataset.discard(exclude)
                            elif idx == 0:
                                dataset.kept_training_session(mask)
                        # label
                        elif test_splitter.option == SplitByType.TRAIL or test_splitter.option == SplitByType.TRAIL_IND:
                            mask, exclude = dataset.pick_trail(mask, num=test_splitter.get_value(), group_idx=group_idx, split_type=test_splitter.get_split_type(), ref_exclude=ref_exclude if (idx == 0) else None)
                            if idx == 0:
                                ref_mask = exclude.copy()
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.option == SplitByType.TRAIL_IND:
                                dataset.discard(exclude)
                        # subject
                        elif test_splitter.option == SplitByType.SUBJECT or test_splitter.option == SplitByType.SUBJECT_IND:
                            mask, exclude = dataset.pick_subject(mask, num=test_splitter.get_value(), group_idx=group_idx, split_type=test_splitter.get_split_type(), ref_exclude=ref_exclude if (idx == 0) else None)
                            if idx == 0:
                                ref_mask = exclude.copy()
                                exclude |= ref_exclude
                            if not mask.any():
                                has_next = False
                                break
                            # independent
                            if test_splitter.option == SplitByType.SUBJECT_IND:
                                dataset.discard(exclude)
                            elif idx == 0:
                                dataset.kept_training_subject(mask)
                        idx += 1

                if not has_next:
                    break
                if idx > 0:
                    dataset.set_test(mask)
                
                # split val data
                mask = dataset.get_remaining()
                idx = 0
                for val_splitter in self.val_splitter_list:
                    if val_splitter.is_option:
                        if self.last_update != checker:
                            return
                        if not val_splitter.is_valid():
                            self.preview_failed = True
                            return
                        # session
                        if val_splitter.option == ValSplitByType.SESSION:
                            mask, exclude = dataset.pick_session(mask, num=val_splitter.get_value(), split_type=val_splitter.get_split_type())
                        # label
                        elif val_splitter.option == ValSplitByType.TRAIL:
                            mask, exclude = dataset.pick_trail(mask, num=val_splitter.get_value(), split_type=val_splitter.get_split_type())
                        # subject
                        elif val_splitter.option == ValSplitByType.SUBJECT:
                            mask, exclude = dataset.pick_subject(mask, num=val_splitter.get_value(), split_type=val_splitter.get_split_type())
                        idx += 1
                if idx > 0:
                    dataset.set_val(mask)
                dataset.set_train()
                if self.last_update != checker:
                    return
                self.datasets.append(dataset)
                group_idx += 1
                if not self.config.is_cross_validation:
                    break
            if self.config.train != TrainingType.IND:
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
            if '-' in self.tree.get_children():
                self.tree.delete('-')
            counter = 0
            while len(self.tree.get_children()) < len(self.datasets):
                if counter > 50:
                    break
                counter += 1
                idx = len(self.tree.get_children())
                dataset = self.datasets[idx]
                self.tree.insert("", 'end', iid=idx, values=('O', dataset.name, sum(dataset.train), sum(dataset.val), sum(dataset.test)))
        self.after(500, self.update_table)

    def show_info(self):
        if len(self.datasets) == 0 or self.tree.focus() == '' or not self.tree.focus().isdigit():
            tk.messagebox.showerror(parent=self, title='Error', message='No valid item is selected')
            return
        idx = int(self.tree.focus())
        target = self.datasets[idx]
        window = DataSplittingInfoWindow(self, target)
        window.get_result()
        # update
        if len(self.datasets) > idx:
            target = self.datasets[idx]
            if self.winfo_exists():
                self.tree.item(idx, values=('O' if target.is_selected else 'X', target.name, sum(target.train), sum(target.val), sum(target.test)))

    def confirm(self):
        # check if dataset is empty
        if self.winfo_ismapped():
            if len(self.datasets) == 0:
                tk.messagebox.showerror(parent=self, title='Error', message='No valid datasets exist')
                return
            # check if data is empty
            for dataset in self.datasets:
                if sum(dataset.test) == 0 or sum(dataset.val) == 0 or sum(dataset.train) == 0:
                    if tk.messagebox.askokcancel(parent=self, title='Warning', message='There are some datasets without training/testing/validation data.\nDo you want to proceed?'):
                        break
                    else:
                        return
        self.withdraw()
        if self.worker.is_alive():
            self.after(1000, self.confirm)
            return
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
        
        train_tree.pack()
        val_tree.pack()
        test_tree.pack()

        select_checkbox.grid(row=0, column=1)
        name_label.grid(row=1, column=0, sticky='e')
        name_entry.grid(row=1, column=1)
        train_frame.grid(row=2, column=0)
        val_frame.grid(row=2, column=1)
        test_frame.grid(row=2, column=2)

        tk.Button(self, text='Save', command=self.confirm).grid(row=3, column=1)
        self.name_var = name_var
        self.select_var = select_var
        self.show_info(train_tree, dataset.train)
        self.show_info(test_tree, dataset.test)
        self.show_info(val_tree, dataset.val)

    def show_info(self, tree, mask):
        data_holder = self.dataset.data_holder
        # traverse subject
        for subject_idx in np.unique( data_holder.subject[mask] ):
            subject_mask = (data_holder.subject == subject_idx) & mask
            subject_root = tree.insert("", 'end', text=f"Subject {data_holder.subject_map[subject_idx]} ({sum(subject_mask)})")
            # traverse session
            for session_idx in np.unique( data_holder.session[subject_mask] ):
                session_mask = (data_holder.session == session_idx) & subject_mask
                session_root = tree.insert(subject_root, 'end', text=f"Session {data_holder.session_map[session_idx]} ({sum(session_mask)})")
                # traverse label
                for label_idx in np.unique( data_holder.label[session_mask] ):
                    label_mask = (data_holder.label == label_idx) & session_mask
                    label_root = tree.insert(session_root, 'end', text=f"Label {data_holder.label_map[label_idx]} ({sum(label_mask)})")
                    # traverse index
                    idx_list = data_holder.idx[label_mask]
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