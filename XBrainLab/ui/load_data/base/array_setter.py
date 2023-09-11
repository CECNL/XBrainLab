import os
import tkinter as  tk
import tkinter.ttk as ttk

from ...base import TopWindow, ValidateException
from .shape_option import RawShapeOtion, EpochShapeOtion, generate_perm

from . import DataType

class ArrayInfoSetter(TopWindow):
    def __init__(self, parent, fp, loaded_array, raw_info, type_ctrl):
        # ==== inits
        super().__init__(parent, "Select Field")
        self.loaded_array = loaded_array
        self.raw_info = raw_info
        self.ret_key = None
        if type_ctrl == DataType.RAW.value:
            OPTION = RawShapeOtion
        else:
            OPTION = EpochShapeOtion
        
        # generate options
        shape_option_perm, shape_option_list = generate_perm(OPTION)

        # generate vars
        self.shape_type_trace = tk.StringVar(self)
        self.data_shape_view = tk.StringVar(self)

        ##
        self.shape_type_trace.set(shape_option_list[0])
        self.shape_type_trace.trace_add('write', self._shape_view_update)
        
        # ======== data key
        data_key_frame = ttk.LabelFrame(self, text="Select shape type")
        tk.Label(data_key_frame, text="Shape type: ").grid(
            row=1, column=0, sticky='w'
        )
        tk.OptionMenu(data_key_frame, self.shape_type_trace, *shape_option_list).grid(
            row=1, column=1, sticky='w'
        )

        tk.Label(data_key_frame, text="Value shape: ").grid(
            row=2, column=0, sticky='w'
        )
        tk.Label(data_key_frame, textvariable=self.data_shape_view).grid(
            row=2, column=1
        )
        
        # ==== sampling rate & channel
        self.sfreq_trace = tk.StringVar(self)
        self.tmin_trace = tk.StringVar(self)
        self.nchan_trace = tk.IntVar(self)
        self.time_trace = tk.IntVar(self)
        self.tmin_trace.set(0)

        attr_frame = tk.Frame(self)
        tk.Label(attr_frame, text='Sampling rate').grid(row=0, column=0, sticky='w')
        if self.raw_info.get_sfreq():
            self.sfreq_trace.set(self.raw_info.get_sfreq())
            tk.Label(attr_frame, textvariable=self.sfreq_trace).grid(
                row=0, column=1, sticky='w'
            )
        else:
            tk.Entry(attr_frame, textvariable=self.sfreq_trace).grid(
                row=0, column=1, sticky='w')   
             
        if type_ctrl == DataType.EPOCH.value:
            tk.Label(attr_frame, text='tmin').grid(row=1, column=0, sticky='w')
            if self.raw_info.get_tmin():
                self.tmin_trace.set(self.raw_info.get_tmin())
                tk.Label(attr_frame, textvariable=self.tmin_trace).grid(
                    row=1, column=1, sticky='w'
                )
            else:
                tk.Entry(attr_frame, textvariable=self.tmin_trace).grid(
                    row=1, column=1, sticky='w')   
                 
        tk.Label(attr_frame, text='channel').grid(
            row=2, column=0, sticky='w'
        )
        tk.Label(attr_frame, textvariable=self.nchan_trace).grid(
            row=2, column=1, sticky='w'
        )
        tk.Label(attr_frame, text='time').grid(
            row=3, column=0, sticky='w'
        )
        tk.Label(attr_frame, textvariable=self.time_trace).grid(
            row=3, column=1, sticky='w'
        )

        # pack
        tk.Label(self, text="Filename: " + os.path.basename(fp)).pack(
            padx=5, pady=10, expand=True
        )
        data_key_frame.pack(padx=5, pady=10, expand=True)
        attr_frame.pack(padx=5, pady=10, expand=True)
        tk.Button(self, text="Confirm", command=self.confirm).pack(
            padx=5, pady=10, expand=True
        )

        # init
        self.OPTION = OPTION
        self.shape_option_perm = shape_option_perm
        self.shape_option_list = shape_option_list
        self._shape_view_update()

    def _shape_view_update(self, *args): # on spinbox change
        data = self.loaded_array
        self.data_shape_view.set(str(data.shape))
        if len(data.shape) != len(self.OPTION):    
            self.nchan_trace.set(-1)
            self.time_trace.set(-1)
        else:
            shape_idx = self.shape_option_perm[
                self.shape_option_list.index( self.shape_type_trace.get() ) 
            ]
            ch_idx = shape_idx.index( self.OPTION.CH )
            time_idx = shape_idx.index( self.OPTION.TIME )
            self.nchan_trace.set( data.shape[ch_idx] )
            self.time_trace.set( data.shape[time_idx] )

    def confirm(self):
        # check attr
        try:
            sfreq = float(self.sfreq_trace.get())
            assert sfreq > 0
        except (ValueError, AssertionError):
            raise ValidateException(self, "Invalid sampling rate.")
        
        # check data
        ## check dim
        data = self.loaded_array
        if len(data.shape) != len(self.OPTION):
            raise ValidateException(
                self, 
                f"Invalid data dimension, should be ({self.shape_type_trace.get()})."
            )
        nchan = self.nchan_trace.get()
        ntimes = self.time_trace.get()
        # check tmin if type is epoch
        tmin = None
        if EpochShapeOtion.EPOCH in self.OPTION:
            try:
                tmin = float(self.tmin_trace.get())
            except ValueError:
                raise ValidateException(self, "Invalid tmin.")
        
        self.raw_info.set_attr(sfreq, nchan, ntimes, tmin)
        shape_idx = self.shape_option_perm[
            self.shape_option_list.index( self.shape_type_trace.get() ) 
        ]
        self.raw_info.set_shape_idx(shape_idx, self.OPTION)
        
        self.ret_key = self.raw_info
        self.destroy()

    def _get_result(self):
        return self.ret_key
