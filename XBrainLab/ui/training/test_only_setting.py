import tkinter as tk

from XBrainLab.training import TestOnlyOption, parse_device_name

from ..base import TopWindow, ValidateException
from ..script import Script
from .training_setting import SetDeviceWindow


class TestOnlySettingWindow(TopWindow):
    __test__ = False # Not a test case
    def __init__(self, parent):
        super().__init__(parent, 'Test Only Setting')
        self.training_option = None
        self.output_dir = None
        self.use_cpu = None
        self.gpu_idx = None
        self.script_history = None

        tk.Label(self, text='Batch size').grid(row=1, column=0)
        tk.Label(self, text='device').grid(row=4, column=0)
        tk.Label(self, text='Output Directory').grid(row=5, column=0)

        bs_entry = tk.Entry(self)
        dev_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_device).grid(row=4, column=2)
        output_dir_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_output_dir).grid(row=5, column=2)

        bs_entry.grid(row=1, column=1, sticky='EW')
        dev_label.grid(row=4, column=1, sticky='EW')
        output_dir_label.grid(row=5, column=1, sticky='EW')
        tk.Button(self, text='Confirm', command=self.confirm).grid(
            row=9, column=0, columnspan=3
        )
        self.columnconfigure([1], weight=1)
        self.rowconfigure(list(range(10)), weight=1)

        self.output_dir_label = output_dir_label
        self.bs_entry = bs_entry
        self.dev_label = dev_label

    def set_device(self):
        setter = SetDeviceWindow(self)
        self.use_cpu, self.gpu_idx = setter.get_result()
        if not self.window_exist:
            return
        self.dev_label.config(text=parse_device_name(self.use_cpu, self.gpu_idx))

    def set_output_dir(self):
        filepath = tk.filedialog.askdirectory(parent=self)
        if filepath:
            self.output_dir = filepath
            self.output_dir_label.config(text=filepath)

    def confirm(self):
        try:
            self.training_option = TestOnlyOption(self.output_dir,
                                    self.use_cpu, self.gpu_idx,
                                    self.bs_entry.get())
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e

        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.training import TestOnlyOption")
        self.script_history.add_import("import torch")

        self.script_history.add_cmd(
            f"output_dir={self.training_option.output_dir!r}"
        )
        self.script_history.add_cmd(f"use_cpu={self.training_option.use_cpu!r}")
        self.script_history.add_cmd(f"gpu_idx={self.training_option.gpu_idx!r}")
        self.script_history.add_cmd(f"bs={self.training_option.bs!r}")

        self.script_history.add_cmd(
            "training_option = TestOnlyOption(output_dir=output_dir, "
        )
        self.script_history.add_cmd("use_cpu=use_cpu, gpu_idx=gpu_idx, ")
        self.script_history.add_cmd("bs=bs)")

        self.destroy()

    def _get_result(self):
        return self.training_option

    def _get_script_history(self):
        return self.script_history
