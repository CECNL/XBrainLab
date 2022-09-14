import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from base.top_window import TopWindow
from .data_holder import DataSet
from enum import Enum
import numpy as np
import time
import threading

VAL_SPLIT_OPTION_NUM = 1
TEST_SPLIT_OPTION_NUM = 3

class DataSplittingSettingWindow(TopWindow):
    def __init__(self, parent, data_holder):
        super().__init__(parent, 'Data splitting')
        self.data_holder = data_holder
        # start of options
        ## training type
        training_type_label = tk.Label(self, text='Training Type')

        training_type_list = [i.value for i in TrainingType]
        training_type_var = tk.StringVar(name='train')
        training_type_var.set(training_type_list[0])
        training_type_option = tk.OptionMenu(self, training_type_var, *training_type_list)

        validation_var_list = []
        
        ## validation/test type
        validation_label = tk.Label(self, text='Validation Set')
        testing_label = tk.Label(self, text='Testing Set')
        test_split_by_list = [i.value for i in SplitByType]
        val_split_by_list = [i.value for i in ValSplitByType]
        validation_option_list = []
        testing_option_list = []
        validation_var_list = []
        testing_var_list = []
        
        for name, option_list, var_list, split_by_list, opt_num in zip(
                                ['Validation',              'Testing'], 
                                [validation_option_list,    testing_option_list], 
                                [validation_var_list,       testing_var_list],
                                [val_split_by_list,         test_split_by_list],
                                [VAL_SPLIT_OPTION_NUM,      TEST_SPLIT_OPTION_NUM]
                            ):
            for i in range(opt_num):
                var = tk.StringVar(name=f'{name}_{i}')
                var.set(split_by_list[0])
                option = tk.OptionMenu(self, var, *split_by_list)
                var_list.append(var)
                option_list.append(option)
        cross_validation_var = tk.BooleanVar(self)
        cross_validation_check_button = tk.Checkbutton(self, text='Cross Validation', var=cross_validation_var)

        ## register callback
        training_type_var.trace_add('write', self.option_menu_callback)
        [validation_var.trace_add('write', self.option_menu_callback) for validation_var in validation_var_list]
        [testing_var.trace_add('write', self.option_menu_callback) for testing_var in testing_var_list]
        # end of options

        # Preview canvas
        canvas_frame = tk.Frame(self)
        canvas = tk.Canvas(canvas_frame)
        legend_frame = tk.Frame(canvas_frame)

        tk.Label(legend_frame, width=2, height=1, bg=DrawColor.TRAIN.value).pack(side='left', padx=10)
        tk.Label(legend_frame, text='Training').pack(side='left')
        tk.Label(legend_frame, width=2, height=1, bg=DrawColor.VAL.value).pack(side='left', padx=(30, 10))
        tk.Label(legend_frame, text='Validation').pack(side='left')
        tk.Label(legend_frame, width=2, height=1, bg=DrawColor.TEST.value).pack(side='left', padx=(30, 10))
        tk.Label(legend_frame, text='Testing').pack(side='left')
        canvas.pack()
        legend_frame.pack()
        
        # btn
        confirm_btn = tk.Button(self, text='Confirm', command=self.confirm)

        # pack
        training_type_label.grid(column=1, row=0, padx=10, pady=(10, 0))
        training_type_option.grid(column=1, row=1, padx=10)        
        inc = 0
        testing_label.grid(column=1, row=2+inc, pady=(10, 0))
        cross_validation_check_button.grid(row=3+inc, column=1)
        inc += 1
        for i in range(TEST_SPLIT_OPTION_NUM):
            testing_option_list[i].grid(column=1, row=3+inc)
            if i > 0:
                testing_option_list[i].grid_remove()
            inc += 1
        validation_label.grid(column=1, row=4+inc, pady=(10, 0))
        for i in range(VAL_SPLIT_OPTION_NUM):
            validation_option_list[i].grid(column=1, row=5+inc)
            if i > 0:
                validation_option_list[i].grid_remove()
            inc += 1
        confirm_btn.grid(column=0, row=6+inc, columnspan=2, pady=(20, 15))
        canvas_frame.grid(column=0, row=0, rowspan=6+inc, pady=(10, 0))
        
        # init member
        self.training_type_var = training_type_var
        self.validation_var_list = validation_var_list
        self.validation_option_list = validation_option_list
        self.testing_var_list = testing_var_list
        self.testing_option_list = testing_option_list
        self.cross_validation_var = cross_validation_var
        self.subject_num = 5
        self.session_num = 5
        self.train_region = DrawRegion(self.session_num, self.subject_num)
        self.val_region = DrawRegion(self.session_num, self.subject_num)
        self.test_region = DrawRegion(self.session_num, self.subject_num)
        self.step2_window = None
        ## canvas
        self.canvas = canvas

        ## init func
        self.option_menu_callback(None,None,None)
        self.draw_preview()

    def option_menu_callback(self, var, index, mode):
        # check disable visibility
        for var_list, option_list in zip([self.validation_var_list, self.testing_var_list], [self.validation_option_list, self.testing_option_list]):
            opt_num = len(option_list)
            for i in range(opt_num):
                if var_list[i].get() == SplitByType.DISABLE.value:
                    if i + 1 < opt_num and option_list[i + 1].winfo_ismapped():
                        self.after(100, lambda idx=i+1,v=var_list: v[idx].set(SplitByType.DISABLE.value))
                        self.after(100, lambda idx=i+1,v=option_list: v[idx].grid_remove())
                        return
                else:
                    if i + 1 < opt_num and not option_list[i + 1].winfo_ismapped():
                        self.after(100, lambda idx=i+1,v=var_list: v[idx].set(SplitByType.DISABLE.value))
                        self.after(100, lambda idx=i+1,v=option_list: v[idx].grid())
                        return
        # reset region
        self.train_region = DrawRegion(self.session_num, self.subject_num)
        self.val_region = DrawRegion(self.session_num, self.subject_num)
        self.test_region = DrawRegion(self.session_num, self.subject_num)
        # handle data
        if self.training_type_var.get() == TrainingType.FULL.value:
            self.handle_data(TrainingType.FULL)
        elif self.training_type_var.get() == TrainingType.IND.value:
            self.handle_data(TrainingType.IND)
        self.draw_preview()

    def handle_data(self, training_type):
        # set init region
        if training_type == TrainingType.FULL:
            self.train_region.set_to(self.session_num, self.subject_num)
        elif training_type == TrainingType.IND:
            self.train_region.set_to(self.session_num, 1)
        # testing
        for idx, testing_var in enumerate(self.testing_var_list):
            # reference region
            if idx == 0:
                ref = self.train_region
            else:
                ref = self.test_region
            # session
            if testing_var.get() == SplitByType.SESSION.value or testing_var.get() == SplitByType.SESSION_IND.value:
                # independent, remove last target
                if testing_var.get() == SplitByType.SESSION_IND.value:
                    tmp = DrawRegion(self.train_region.w, self.train_region.h)
                    tmp.copy(ref)
                    tmp.set_to(ref.to_x - 1, ref.to_y)
                self.test_region.set_from(ref.to_x - 1, ref.from_y)
                self.test_region.set_to(ref.to_x, ref.to_y)
                if testing_var.get() == SplitByType.SESSION_IND.value:
                    self.train_region.mask(tmp)
                # avoid being used by validation set
                if idx == 0:
                    self.train_region.to_x -= 1
            # label
            elif testing_var.get() == SplitByType.TRAIL.value or testing_var.get() == SplitByType.TRAIL_IND.value:
                # independent, remove last target
                if testing_var.get() == SplitByType.TRAIL_IND.value:
                    tmp = DrawRegion(ref.w, ref.h)
                    tmp.copy(ref)
                    tmp.set_w(0, ref.from_w + (ref.to_w - ref.from_w) * 0.8)
                self.test_region.copy(ref)
                self.test_region.set_w(self.test_region.from_w + (self.test_region.to_w - self.test_region.from_w) * 0.8, self.test_region.to_w)
                if testing_var.get() == SplitByType.TRAIL_IND.value:
                    self.train_region.mask(tmp)
            # subject
            elif testing_var.get() == SplitByType.SUBJECT.value or testing_var.get() == SplitByType.SUBJECT_IND.value:
                # independent, remove last target
                if testing_var.get() == SplitByType.SUBJECT_IND.value:
                    tmp = DrawRegion(self.train_region.w, self.train_region.h)
                    tmp.copy(ref)
                    tmp.set_from(ref.from_x, ref.from_y)
                    tmp.set_to(ref.to_x, ref.to_y - 1)
                self.test_region.set_from(ref.from_x, ref.to_y - 1)
                self.test_region.set_to(ref.to_x, ref.to_y)
                if testing_var.get() == SplitByType.SUBJECT_IND.value:
                    self.train_region.mask(tmp)
                # avoid being used by validation set
                if idx == 0:
                    self.train_region.to_y -= 1
        self.train_region.mask(self.test_region)
        # validation
        for idx, validation_var in enumerate(self.validation_var_list):
            if validation_var.get() == ValSplitByType.SESSION.value:
                self.val_region.copy(self.train_region)
                self.val_region.set_from(self.train_region.to_x - 1, self.train_region.from_y)
                self.val_region.set_to(self.train_region.to_x, self.train_region.to_y)
                # fix overlapped with testing set
                if self.val_region.intersect(self.test_region):
                    self.train_region.to_x -= 1
                    self.val_region.set_from(self.train_region.to_x - 1, self.train_region.from_y)
                    self.val_region.set_to(self.train_region.to_x, self.train_region.to_y)
            elif validation_var.get() == ValSplitByType.TRAIL.value:
                self.val_region.copy(self.train_region)
                self.val_region.fix()
                self.val_region.set_w(self.train_region.to_w - 0.2, self.train_region.to_w)
                # fix overlapped with testing set
                if self.val_region.intersect(self.test_region):
                    self.train_region.to_w = self.test_region.from_w
                    self.val_region.set_w(self.train_region.to_w - 0.2, self.train_region.to_w)
                self.val_region.fix_intersect(self.test_region)
            elif validation_var.get() == ValSplitByType.SUBJECT.value:
                self.val_region.copy(self.train_region)
                self.val_region.set_from(self.train_region.from_x, self.train_region.to_y - 1)
                self.val_region.set_to(self.train_region.to_x, self.train_region.to_y)
                # fix overlapped with testing set
                if self.val_region.intersect(self.test_region):
                    self.train_region.to_y -= 1
                    self.val_region.set_from(self.train_region.from_x, self.train_region.to_y - 1)
                    self.val_region.set_to(self.train_region.to_x, self.train_region.to_y)
        self.val_region.mask_intersect(self.train_region)
        self.train_region.mask(self.val_region)
        
    def draw_preview(self):
        left = 80
        top = 5
        right = bottom = 30
        w = 360
        h = 120
        canvas_width = w + left + right
        canvas_height = h + top + bottom
        subject = self.subject_num
        session = self.session_num
        delta_x = w / session
        delta_y = h / subject
        canvas = self.canvas
        canvas.config(width=canvas_width, height=canvas_height)
        canvas.delete("all")
        # draw area
        for var, color in zip([self.train_region, self.val_region, self.test_region], [DrawColor.TRAIN, DrawColor.VAL, DrawColor.TEST]):
            for i in range(var.w):
                for j in range(var.h):
                    if var.from_canvas[i, j] == var.to_canvas[i, j]:
                        continue
                    canvas.create_rectangle(left + delta_x * (i + var.from_canvas[i, j]), top + delta_y * j, 
                                            left + delta_x * (i + var.to_canvas[i, j]), top + delta_y * (j + 1), 
                                            fill=color.value, width=0)
        # draw box
        canvas.create_rectangle(left, top, left + w, top + h)
        canvas.create_text(left / 2, top + h / 2, text='Subject')
        canvas.create_text(left + w / 2, top + h + bottom / 2, text='Session')

        # draw border
        for i in range(1, subject):
            d = top + h / subject * i
            canvas.create_line(left, d, left + w, d, dash=(4, 4))

        for i in range(1, session):
            d = left + w / session * i
            canvas.create_line(d, top, d, top + h, dash=(4, 4))

    def confirm(self):
        for i in TrainingType:
            if i.value == self.training_type_var.get():
                train = i
        
        val = []
        test = []
        for var_list, by_type, arr in zip(
                        [self.validation_var_list, self.testing_var_list],
                        [ValSplitByType, SplitByType], 
                        [val, test]
                    ):
            for j in var_list:
                for i in by_type:
                    if i.value == j.get():
                        if i == by_type.DISABLE and len(arr) > 0:
                            break
                        arr.append(i)
        config = DataSplittingConfig(train, val, test, self.cross_validation_var.get())

        self.step2_window = DataSplittingWindow(self.master, self.title(), self.data_holder, config)
        self.destroy()

    def _get_result(self):
        try:
            self.step2_window.wait_window()
            return self.step2_window._get_result()
        except:
            return None
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
                            mask, exclude = dataset.pick_session(mask, num=test_splitter.get_value(), skip=0, is_ratio=test_splitter.is_ratio(), ref_exclude=ref_exclude if (idx == 0) else None)
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
                            mask, exclude = dataset.pick_trail(mask, num=test_splitter.get_value(), skip=0, is_ratio=test_splitter.is_ratio(), ref_exclude=ref_exclude if (idx == 0) else None)
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
                            mask, exclude = dataset.pick_subject(mask, num=test_splitter.get_value(), skip=0, is_ratio=test_splitter.is_ratio(), ref_exclude=ref_exclude if (idx == 0) else None)
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
                            mask, exclude = dataset.pick_session(mask, num=val_splitter.get_value(), skip=0, is_ratio=val_splitter.is_ratio())
                        # label
                        elif val_splitter.option == ValSplitByType.TRAIL:
                            mask, exclude = dataset.pick_trail(mask, num=val_splitter.get_value(), skip=0, is_ratio=val_splitter.is_ratio())
                        # subject
                        elif val_splitter.option == ValSplitByType.SUBJECT:
                            mask, exclude = dataset.pick_subject(mask, num=val_splitter.get_value(), skip=0, is_ratio=val_splitter.is_ratio())
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

class TrainingType(Enum):
    FULL = 'Full Data'
    IND = 'Individual'

class SplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    SESSION_IND = 'By Session (Independent)'
    TRAIL = 'By Trail'
    TRAIL_IND = 'By Trail (Independent)'
    SUBJECT = 'By Subject'
    SUBJECT_IND = 'By Subject (Independent)'

class ValSplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    TRAIL = 'By Trail'
    SUBJECT = 'By Subject'


class DrawColor(Enum):
    TRAIN = 'DodgerBlue'
    VAL = 'LightBlue'
    TEST = 'green'

class DrawRegion():
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.from_canvas = np.zeros((w, h))
        self.to_canvas = np.zeros((w, h))

        self.from_x = 0
        self.from_y = 0
        self.to_x = 0
        self.to_y = 0
        self.from_w = 0
        self.to_w = 1

    def set_from(self, x, y):
        self.reset()
        self.from_x = x
        self.from_y = y
    
    def set_to(self, x, y):
        self.to_x = x
        self.to_y = y
        self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = self.from_w
        self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = self.to_w

    def mask(self, rhs):
        # clear masking region
        idx = rhs.from_canvas != rhs.to_canvas

        filter_idx = idx & (self.from_canvas < rhs.from_canvas) & (rhs.from_canvas < self.to_canvas)
        self.to_canvas[ filter_idx ] = rhs.from_canvas[ filter_idx ].copy()
        
        filter_idx = idx & (self.from_canvas <= rhs.to_canvas) & (rhs.to_canvas <= self.to_canvas)
        self.from_canvas[filter_idx] = rhs.to_canvas[ filter_idx ].copy()
        
        filter_idx = idx & (rhs.from_canvas <= self.from_canvas) & (self.to_canvas <= rhs.to_canvas)
        self.to_canvas[filter_idx] = self.from_canvas[filter_idx].copy()
    
    def mask_intersect(self, rhs):
        # remove if rhs is not included
        idx = (rhs.to_canvas != 1) & (rhs.from_canvas != rhs.to_canvas) & (self.from_canvas != self.to_canvas)
        self.to_canvas[idx] = rhs.to_canvas[idx].copy()

        idx = (rhs.from_canvas == rhs.to_canvas) & (self.from_canvas != self.to_canvas)
        self.to_canvas[idx] = self.from_canvas[idx].copy()
        
    def intersect(self, rhs):
        # check if all region is intersected
        idx = self.from_canvas != self.to_canvas
        return ((rhs.from_canvas[idx] != rhs.to_canvas[idx]) & (rhs.from_canvas[idx] == self.from_canvas[idx]) & (rhs.to_canvas[idx] == self.to_canvas[idx])).all()
    
    def fix_intersect(self, rhs):
        for i in range(self.w):
            for j in range(self.h):
                if (rhs.from_canvas[i, j] != rhs.to_canvas[i, j] ) and (rhs.from_canvas[i, j] == self.from_canvas[i, j]) and (rhs.to_canvas[i, j] == self.to_canvas[i, j]):
                    self.from_canvas[i, j] = np.maximum(self.from_canvas[i, j] - 0.2, 0)
                    self.to_canvas[i, j] = np.minimum(self.to_canvas[i, j] - 0.2, rhs.from_canvas[i, j])



    def set_w(self, a, b):
        self.from_w = a
        self.to_w = b
        idx = self.from_canvas != self.to_canvas
        self.from_canvas[idx] = self.from_w
        self.to_canvas[idx] = self.to_w
    
    def reset(self):
        self.from_canvas *= 0
        self.to_canvas *= 0

    def copy(self, rhs):
        self.from_x = rhs.from_x
        self.from_y = rhs.from_y
        self.to_x = rhs.to_x
        self.to_y = rhs.to_y
        self.from_w = rhs.from_w
        self.to_w = rhs.to_w
        self.from_canvas = rhs.from_canvas.copy()
        self.to_canvas = rhs.to_canvas.copy()

    def fix(self):
        self.to_canvas[self.to_x:, :] = self.from_canvas[self.to_x:, :]
        self.to_canvas[:, self.to_y:] = self.from_canvas[:, self.to_y:]
        

class DataSplittingOption():
    def __init__(self, is_option, text, option):
        self.is_option = is_option
        self.option = option
        self.text = text
        self.split_var = None
        self.entry_var = None

        self.is_valid_var = False
        self.value_var = None
        self.is_ratio_var = None
        self.is_KFold_var = None
    
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
        return False

    def _get_value(self):
        if not self._is_valid():
            return 0
        return float(self.entry_var.get())

    def _is_ratio(self):
        if self.split_var is None:
            return False
        return self.split_var.get() == SplitUnit.RATIO.value

    def _is_KFold(self):
        if self.split_var is None:
            return False
        return self.split_var.get() == SplitUnit.KFOLD.value

    def to_thread(self):
        self.is_valid_var = self._is_valid()
        self.value_var = self._get_value()
        self.is_ratio_var = self._is_ratio()
        self.is_KFold_var = self._is_KFold()

    def is_valid(self):
        return self.is_valid_var
    
    def get_value(self):
        return self.value_var
        
    def is_ratio(self):
        return self.is_ratio_var
    
    def is_KFold(self):
        return self.is_KFold_var

    def set_split_var(self, root, val, callback):
        self.split_var = tk.StringVar(root)
        self.split_var.set(val)
        self.split_var.trace_add('write', callback)

    def set_entry_var(self, root, val, callback):
        self.entry_var = tk.StringVar(root)
        self.entry_var.set(val)
        self.entry_var.trace_add('write', callback)

class DataSplittingConfig():
    def __init__(self, train, val, test, is_cross_validation):
        self.train = train
        self.val = val
        self.test = test
        self.is_cross_validation = is_cross_validation
    
    def get_splitter_option(self):
        val = []
        for v in self.val:
            is_option = not (v == ValSplitByType.DISABLE)
            text = v.value
            val.append(DataSplittingOption(is_option, text, option=v))
        test = []
        for t in self.test:
            is_option = not (t == SplitByType.DISABLE)
            text = t.value
            test.append(DataSplittingOption(is_option, text, option=t))

        return val, test
    
###
class SplitUnit(Enum):
    RATIO = 'Ratio'
    NUMBER = 'Number'
    KFOLD = 'K Fold'
###

if __name__ == '__main__':
    import numpy as np

    from data_holder import Epochs
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
    window = DataSplittingSettingWindow(root, data_holder)
    # window = DataSplittingWindow(root, 'Data splitting', data_holder, config)

    print (window.get_result())
    root.destroy()