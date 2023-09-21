import inspect
import tkinter as tk

import torch

from XBrainLab.training import (
    TRAINING_EVALUATION,
    TrainingOption,
    parse_device_name,
    parse_optim_name,
)

from ..base import TopWindow, ValidateException
from ..script import Script
from ..widget import EditableTreeView


class TrainingSettingWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Training Setting')
        self.training_option = None
        self.output_dir = None
        self.optim = None
        self.optim_params = None
        self.use_cpu = None
        self.gpu_idx = None
        self.script_history = None

        tk.Label(self, text='Epoch').grid(row=0, column=0)
        tk.Label(self, text='Batch size').grid(row=1, column=0)
        tk.Label(self, text='Learning rate').grid(row=2, column=0)
        tk.Label(self, text='Optimizer').grid(row=3, column=0)
        tk.Label(self, text='device').grid(row=4, column=0)
        tk.Label(self, text='Output Directory').grid(row=5, column=0)
        tk.Label(self, text='CheckPoint epoch').grid(row=6, column=0)
        tk.Label(self, text='Evaluation').grid(row=7, column=0)
        tk.Label(self, text='Repeat Number').grid(row=8, column=0)

        epoch_entry = tk.Entry(self)
        bs_entry = tk.Entry(self)
        lr_entry = tk.Entry(self)
        opt_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_optimizer).grid(row=3, column=2)
        dev_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_device).grid(row=4, column=2)
        output_dir_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_output_dir).grid(row=5, column=2)
        checkpoint_entry = tk.Entry(self)

        evaluation_var = tk.StringVar(self)
        evaluation_list = [i.value for i in TRAINING_EVALUATION]
        evaluation_var.set(evaluation_list[0])
        evaluation_option = tk.OptionMenu(self, evaluation_var, *evaluation_list)
        repeat_entry = tk.Entry(self)

        epoch_entry.grid(row=0, column=1, sticky='EW')
        bs_entry.grid(row=1, column=1, sticky='EW')
        lr_entry.grid(row=2, column=1, sticky='EW')
        opt_label.grid(row=3, column=1, sticky='EW')
        dev_label.grid(row=4, column=1, sticky='EW')
        output_dir_label.grid(row=5, column=1, sticky='EW')
        checkpoint_entry.grid(row=6, column=1, sticky='EW')
        evaluation_option.grid(row=7, column=1, sticky='EW')
        repeat_entry.grid(row=8, column=1, sticky='EW')
        tk.Button(self, text='Confirm', command=self.confirm).grid(
            row=9, column=0, columnspan=3
        )
        self.columnconfigure([1], weight=1)
        self.rowconfigure(list(range(10)), weight=1)

        self.output_dir_label = output_dir_label
        self.epoch_entry = epoch_entry
        self.bs_entry = bs_entry
        self.lr_entry = lr_entry
        self.checkpoint_entry = checkpoint_entry
        self.dev_label = dev_label
        self.opt_label = opt_label
        self.evaluation_var = evaluation_var
        self.repeat_entry = repeat_entry

    def set_optimizer(self):
        setter = SetOptimizerWindow(self)
        optim, optim_params = setter.get_result()
        if not self.window_exist:
            return
        if optim and optim_params:
            self.optim = optim
            self.optim_params = optim_params
            self.opt_label.config(text=parse_optim_name(optim, optim_params))

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
        evaluation_option = None
        for i in TRAINING_EVALUATION:
            if i.value == self.evaluation_var.get():
                evaluation_option = i

        try:
            self.training_option = TrainingOption(
                self.output_dir, self.optim, self.optim_params,
                self.use_cpu, self.gpu_idx,
                self.epoch_entry.get(),
                self.bs_entry.get(),
                self.lr_entry.get(),
                self.checkpoint_entry.get(),
                evaluation_option,
                self.repeat_entry.get()
            )
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e

        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.training import TrainingOption")
        self.script_history.add_import(
            "from XBrainLab.training import TRAINING_EVALUATION"
        )
        self.script_history.add_import("import torch")

        self.script_history.add_cmd(
            f"output_dir={self.training_option.output_dir!r}"
        )
        self.script_history.add_cmd(f"optim=torch.optim.{self.optim.__name__}")
        self.script_history.add_cmd(
            f"optim_params={self.training_option.optim_params!r}"
        )
        self.script_history.add_cmd(f"use_cpu={self.training_option.use_cpu!r}")
        self.script_history.add_cmd(f"gpu_idx={self.training_option.gpu_idx!r}")
        self.script_history.add_cmd(f"epoch={self.training_option.epoch!r}")
        self.script_history.add_cmd(f"bs={self.training_option.bs!r}")
        self.script_history.add_cmd(f"lr={self.training_option.lr!r}")
        self.script_history.add_cmd(
            f"checkpoint_epoch={self.training_option.checkpoint_epoch!r}"
        )
        self.script_history.add_cmd(
            f"evaluation_option={self.training_option.get_evaluation_option_repr()}"
        )
        self.script_history.add_cmd(
            f"repeat_num={self.training_option.repeat_num!r}"
        )

        self.script_history.add_cmd(
            "training_option = TrainingOption(output_dir=output_dir, "
        )
        self.script_history.add_cmd("optim=optim, optim_params=optim_params, ")
        self.script_history.add_cmd("use_cpu=use_cpu, gpu_idx=gpu_idx, ")
        self.script_history.add_cmd("epoch=epoch, ")
        self.script_history.add_cmd("bs=bs, ")
        self.script_history.add_cmd("lr=lr, ")
        self.script_history.add_cmd("checkpoint_epoch=checkpoint_epoch, ")
        self.script_history.add_cmd("evaluation_option=evaluation_option, ")
        self.script_history.add_cmd("repeat_num=repeat_num)")

        self.destroy()

    def _get_result(self):
        return self.training_option

    def _get_script_history(self):
        return self.script_history
##
class SetOptimizerWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Optimizer Setting')
        self.optim = None
        self.optim_params = None

        algo_label = tk.Label(self, text='Algorithm')

        algo_map = {
            c[0]: c[1] for c in inspect.getmembers(torch.optim, inspect.isclass)
        }
        selected_algo_name = tk.StringVar(self)
        selected_algo_name.trace('w', self.on_algo_select)
        alg_opt = tk.OptionMenu(self, selected_algo_name, *algo_map)

        params_frame = tk.LabelFrame(self, text='Parameters')
        columns = ('param', 'value')
        params_tree = EditableTreeView(
            params_frame, editableCols=['#2'], columns=columns, show='headings'
        )
        params_tree.pack(fill=tk.BOTH, expand=True)

        PADDING = 5

        algo_label.grid(row=0, column=0, sticky='E', padx=PADDING)
        alg_opt.grid(row=0, column=1, sticky='W', padx=PADDING)
        params_frame.grid(row=1, column=0, columnspan=2, sticky='NEWS')
        tk.Button(self, text='Confirm', command=self.confirm).grid(
            row=2, column=0, columnspan=2, pady=PADDING
        )


        self.algo_map = algo_map
        self.selected_algo_name = selected_algo_name
        self.params_tree = params_tree

        selected_algo_name.set(next(iter(algo_map.keys())))
        if False: # test code
            selected_algo_name.set('Adam') # test code

    def on_algo_select(self, *args):
        target = self.algo_map[self.selected_algo_name.get()]
        self.params_tree.delete(*self.params_tree.get_children())
        if target:
            sigs = inspect.signature(target.__init__)
            params = list(sigs.parameters)[2:] # skip self and lr
            for param in params:
                if 'lr' in param:
                    continue
                if (
                    sigs.parameters[param].default == inspect._empty
                    or sigs.parameters[param].default is None
                ):
                    value = ''
                else:
                    value = sigs.parameters[param].default
                    if False: # test code
                        if param == 'weight_decay':
                            value = 0.1
                self.params_tree.insert('', index='end', values=(param, value))

    def confirm(self):
        self.params_tree.submitEntry()
        optim_params = {}
        target = self.algo_map[self.selected_algo_name.get()]
        reason = None
        try:
            for item in self.params_tree.get_children():
                param, value = self.params_tree.item(item)['values']
                if value != '':
                    if isinstance(value, str) and len(value.split()) > 1:
                        value = [float(v) for v in value.split()]
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    else:
                        value = float(value)
                    optim_params[param] = value
            target([torch.Tensor()], lr=1, **optim_params)
        except Exception:
            reason = 'Invalid parameter'

        if reason:
            raise ValidateException(window=self, message=reason)

        self.optim_params = optim_params
        self.optim = target

        self.destroy()

    def _get_result(self):
        return self.optim, self.optim_params
##
class SetDeviceWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Device Setting')
        self.use_cpu = None
        self.gpu_idx = None

        device_list = tk.Listbox(self, selectmode=tk.SINGLE, exportselection=False)
        device_list.insert(tk.END, 'CPU')
        for i in range(torch.cuda.device_count()):
            name = f'{i} - {torch.cuda.get_device_name(i)}'
            device_list.insert(tk.END, name)
        device_list.select_set(torch.cuda.device_count())

        PADDING = 5
        device_list.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)
        tk.Button(self, text='Confirm', command=self.confirm).pack(pady=PADDING)

        self.device_list = device_list

    def confirm(self):
        result = next(iter(self.device_list.curselection()))
        self.use_cpu = result == 0
        if result > 0:
            self.gpu_idx = result - 1
        self.destroy()

    def _get_result(self):
        return self.use_cpu, self.gpu_idx
