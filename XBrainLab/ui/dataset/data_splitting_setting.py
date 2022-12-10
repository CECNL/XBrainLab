import numpy as np
from enum import Enum
import tkinter as tk

from ..base import TopWindow, InitWindowValidateException
from .data_splitting import DataSplittingWindow, DataSplittingConfigHolder

from XBrainLab.dataset import Epochs
from XBrainLab.dataset import TrainingType, SplitByType, ValSplitByType, SplitUnit

class DataSplittingSettingWindow(TopWindow):
    def __init__(self, parent, epoch_data):
        super().__init__(parent, 'Data splitting')
        self.epoch_data = epoch_data
        # validate input data
        self.check_data()
        # start of options
        ## training type
        training_type_label = tk.Label(self, text='Training Type')
        training_type_list = [i.value for i in TrainingType]
        training_type_var = tk.StringVar(self)
        training_type_var.set(training_type_list[0])
        training_type_option = tk.OptionMenu(self, training_type_var, *training_type_list)
        ## validation type
        validation_label = tk.Label(self, text='Validation Set')
        VAL_SPLIT_OPTION_NUM = 1
        val_split_by_list = [i.value for i in ValSplitByType]
        validation_var_list = []
        validation_option_list = []
        self.init_option_list(VAL_SPLIT_OPTION_NUM, val_split_by_list, validation_var_list, validation_option_list)
        ## test type
        testing_label = tk.Label(self, text='Testing Set')
        TEST_SPLIT_OPTION_NUM = 3
        test_split_by_list = [i.value for i in SplitByType]
        testing_var_list = []
        testing_option_list = []
        self.init_option_list(TEST_SPLIT_OPTION_NUM, test_split_by_list, testing_var_list, testing_option_list)
        ## cross validation
        cross_validation_var = tk.BooleanVar(self)
        cross_validation_check_button = tk.Checkbutton(self, text='Cross Validation', var=cross_validation_var)

        ## register callback
        training_type_var.trace_add('write', self.option_menu_callback)
        [validation_var.trace_add('write', self.option_menu_callback) for validation_var in validation_var_list]
        [testing_var.trace_add('write', self.option_menu_callback) for testing_var in testing_var_list]

        # Preview canvas
        canvas_frame = tk.Frame(self)
        ## canvas_frame
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
        ## test options
        testing_label.grid(column=1, row=2+inc, pady=(10, 0))
        cross_validation_check_button.grid(row=3+inc, column=1)
        for i in range(TEST_SPLIT_OPTION_NUM):
            testing_option_list[i].grid(column=1, row=4+inc)
            # hide disabled
            if i > 0:
                testing_option_list[i].grid_remove()
            inc += 1
        ## validation options
        validation_label.grid(column=1, row=4+inc, pady=(10, 0))
        for i in range(VAL_SPLIT_OPTION_NUM):
            validation_option_list[i].grid(column=1, row=5+inc)
            # hide disabled
            if i > 0:
                validation_option_list[i].grid_remove()
            inc += 1
        confirm_btn.grid(column=0, row=5+inc, columnspan=2, pady=(20, 15))
        canvas_frame.grid(column=0, row=0, rowspan=5+inc, pady=(10, 0))
        
        # init member
        self.subject_num = 5
        self.session_num = 5
        ## var
        self.training_type_var = training_type_var
        self.testing_var_list = testing_var_list
        self.validation_var_list = validation_var_list
        self.cross_validation_var = cross_validation_var
        ## option
        self.testing_option_list = testing_option_list
        self.validation_option_list = validation_option_list
        ## region info
        self.train_region = DrawRegion(self.session_num, self.subject_num)
        self.val_region = DrawRegion(self.session_num, self.subject_num)
        self.test_region = DrawRegion(self.session_num, self.subject_num)
        ## canvas
        self.canvas = canvas
        ## step 2
        self.step2_window = None
        
        # init func
        self.option_menu_callback()
        self.draw_preview()

    def check_data(self):
        if type(self.epoch_data) != Epochs:
            raise InitWindowValidateException(self, 'No valid epoch data is generated')

    def init_option_list(self, opt_num, split_by_list, var_list, option_list):
        for i in range(opt_num):
            var = tk.StringVar(self)
            var.set(split_by_list[0])
            option = tk.OptionMenu(self, var, *split_by_list)
            var_list.append(var)
            option_list.append(option)
    #
    def check_option_disable_status(self, var_list, option_list):
        opt_num = len(option_list)
        for i in range(opt_num):
            if var_list[i].get() == SplitByType.DISABLE.value:
            # disable next level
                if i + 1 < opt_num and option_list[i + 1].winfo_ismapped():
                    self.after(100, lambda win=self,idx=i+1,v=var_list: v[idx].set(SplitByType.DISABLE.value))
                    self.after(100, lambda win=self,idx=i+1,v=option_list: v[idx].grid_remove())
                    return True
            else:
            # enable next level
                if i + 1 < opt_num and not option_list[i + 1].winfo_ismapped():
                    self.after(100, lambda win=self,idx=i+1,v=var_list: v[idx].set(SplitByType.DISABLE.value))
                    self.after(100, lambda win=self,idx=i+1,v=option_list: v[idx].grid())
                    return True
        return False

    def option_menu_callback(self, *args):
        # check disable visibility
        if self.check_option_disable_status(self.testing_var_list, self.testing_option_list):
            return
        if self.check_option_disable_status(self.validation_var_list, self.validation_option_list):
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
        self.fix_window_size()
        self.draw_preview()
    #
    def handle_data(self, training_type):
        # set init region based on training type
        if training_type == TrainingType.FULL:
            self.train_region.set_to(self.session_num, self.subject_num, from_w=0, to_w=1)
        elif training_type == TrainingType.IND:
            self.train_region.set_to(self.session_num, y=1, from_w=0, to_w=1)
        #
        self.handle_testing()
        self.train_region.mask(self.test_region)
        
        self.handle_validation()
        self.train_region.mask(self.val_region)
    
    def handle_testing(self):
        for idx, testing_var in enumerate(self.testing_var_list):
            # reference region
            ref = DrawRegion(self.train_region.w, self.train_region.h)
            if idx == 0:
                ref.copy(self.train_region)
            else:
                ref.copy(self.test_region)
            # session
            if testing_var.get() == SplitByType.SESSION.value or testing_var.get() == SplitByType.SESSION_IND.value:
                # independent, remove last target
                is_independent = testing_var.get() == SplitByType.SESSION_IND.value
                if is_independent:
                    tmp = DrawRegion(self.train_region.w, self.train_region.h)
                    tmp.copy(ref)
                    tmp.change_to(ref.to_x - 1, ref.to_y)
                self.test_region.set_from(ref.to_x - 1, ref.from_y)
                self.test_region.set_to_ref(ref.to_x, ref.to_y, ref)
                if is_independent:
                    self.train_region.mask(tmp)
            # label
            elif testing_var.get() == SplitByType.TRIAL.value or testing_var.get() == SplitByType.TRIAL_IND.value:
                # independent, remove last target
                is_independent = testing_var.get() == SplitByType.TRIAL_IND.value
                if is_independent:
                    tmp = DrawRegion(ref.w, ref.h)
                    tmp.copy(ref)
                    tmp.decrease_w_tail(0.8)
                self.test_region.copy(ref)
                self.test_region.decrease_w_head(0.8)
                if is_independent:
                    self.train_region.mask(tmp)
            # subject
            elif testing_var.get() == SplitByType.SUBJECT.value or testing_var.get() == SplitByType.SUBJECT_IND.value:
                # independent, remove last target
                is_independent = testing_var.get() == SplitByType.SUBJECT_IND.value
                if is_independent:
                    tmp = DrawRegion(self.train_region.w, self.train_region.h)
                    tmp.copy(ref)
                    tmp.change_to(ref.to_x, ref.to_y - 1)
                self.test_region.set_from(ref.from_x, ref.to_y - 1)
                self.test_region.set_to_ref(ref.to_x, ref.to_y, ref)
                if is_independent:
                    self.train_region.mask(tmp)
                # avoid being used by validation

    def handle_validation(self):
        for idx, validation_var in enumerate(self.validation_var_list):
            if validation_var.get() == ValSplitByType.SESSION.value:
                self.val_region.copy(self.train_region)
                self.val_region.set_from(self.train_region.to_x - 1, self.train_region.from_y)
                self.val_region.set_to_ref(self.train_region.to_x, self.train_region.to_y, self.train_region)
            elif validation_var.get() == ValSplitByType.TRIAL.value:
                self.val_region.copy(self.train_region)
                self.val_region.decrease_w_head(0.8)
            elif validation_var.get() == ValSplitByType.SUBJECT.value:
                self.val_region.copy(self.train_region)
                self.val_region.set_from(self.train_region.from_x, self.train_region.to_y - 1)
                self.val_region.set_to_ref(self.train_region.to_x, self.train_region.to_y, self.train_region)
    #
    def draw_preview(self):
        # preview region padding
        left = 80
        top = 5
        right = bottom = 30
        # preview region size
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
            for i in range(var.from_x, var.to_x):
                for j in range(var.from_y, var.to_y):
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
    #
    def retreive_result(self, var_list, ByType, arr):
        for j in var_list:
                for i in ByType:
                    if i.value == j.get():
                        if i == ByType.DISABLE and len(arr) > 0:
                            break
                        arr.append(i)
    
    def confirm(self):
        # get training type
        for i in TrainingType:
            if i.value == self.training_type_var.get():
                train_type = i
        # get training type
        val_type_list = []
        test_type_list = []
        self.retreive_result(self.validation_var_list, ValSplitByType, val_type_list)
        self.retreive_result(self.testing_var_list, SplitByType, test_type_list)
        config = DataSplittingConfigHolder(train_type, val_type_list, test_type_list, is_cross_validation=self.cross_validation_var.get())

        self.step2_window = DataSplittingWindow(self.master, self.title(), self.epoch_data, config)
        self.destroy()
    #
    def _get_result(self):
        try:
            return self.step2_window.get_result()
        except:
            return None

    def _get_script_history(self):
        try:
            return self.step2_window.get_script_history()
        except:
            return None

###

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
        # global parms
        self.from_x = 0
        self.from_y = 0
        self.to_x = 0
        self.to_y = 0

    def reset(self):
        self.from_canvas *= 0
        self.to_canvas *= 0
        
    def set_from(self, x, y):
        self.reset()
        self.from_x = x
        self.from_y = y
    
    def set_to_ref(self, x, y, ref):
        self.to_x = x
        self.to_y = y
        self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = ref.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y]
        self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = ref.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y]
    
    def set_to(self, x, y, from_w, to_w):
        self.to_x = x
        self.to_y = y
        self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = from_w
        self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = to_w
    
    def change_to(self, x, y):
        self.to_x = x
        self.to_y = y

    def mask(self, rhs):
        # clear masking region
        idx = rhs.from_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y] != rhs.to_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y]
        filter_idx = idx & (self.from_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y] <= rhs.from_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y]) & (rhs.from_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y] <= self.to_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y])
        
        self.to_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y] *= np.logical_not(filter_idx)
        self.to_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y] += filter_idx * rhs.from_canvas[rhs.from_x:rhs.to_x, rhs.from_y:rhs.to_y]
        if (self.to_canvas[ self.to_x - 1 , self.from_y:self.to_y ] == self.from_canvas[ self.to_x - 1 , self.from_y:self.to_y ]).all():
            self.to_x -= 1
        if (self.to_canvas[ self.from_x:self.to_x, self.to_y - 1 ] == self.from_canvas[ self.from_x:self.to_x, self.to_y - 1]).all():
            self.to_y -= 1
    
    def decrease_w_tail(self, w):
        self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = (self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] - self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y]) * w + self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y]

    def decrease_w_head(self, w):
        self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y] = (self.to_canvas[self.from_x:self.to_x, self.from_y:self.to_y] - self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y]) * w + self.from_canvas[self.from_x:self.to_x, self.from_y:self.to_y]

    def copy(self, rhs):
        self.from_x = rhs.from_x
        self.from_y = rhs.from_y
        self.to_x = rhs.to_x
        self.to_y = rhs.to_y
        self.from_canvas = rhs.from_canvas.copy()
        self.to_canvas = rhs.to_canvas.copy()
