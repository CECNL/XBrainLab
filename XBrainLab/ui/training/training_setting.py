import tkinter as tk
import tkinter.filedialog
import inspect
import torch

from ..base import TopWindow, ValidateException
from ..widget import EditableTreeView
from ..script import Script
from XBrainLab.training import TrainingOption, TRAINING_EVALUATION, parse_device_name, parse_optim_name

class TrainingSettingWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Training Setting')
        self.training_option = None
        self.output_dir = None
        self.optim = None
        self.optim_parms = None
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
        tk.Button(self, text='set', command=self.set_optimizer).grid(row=3,column=2)
        dev_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_device).grid(row=4,column=2)
        output_dir_label = tk.Label(self)
        tk.Button(self, text='set', command=self.set_output_dir).grid(row=5,column=2)
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
        tk.Button(self, text='Confirm', command=self.confirm).grid(row=9, column=0, columnspan=3)
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
        optim, optim_parms = setter.get_result()
        if not self.window_exist:
            return
        if optim and optim_parms:
            self.optim = optim
            self.optim_parms = optim_parms
            self.opt_label.config(text=parse_optim_name(optim, optim_parms))
        
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
            self.training_option = TrainingOption(self.output_dir, self.optim, self.optim_parms, 
                                    self.use_cpu, self.gpu_idx, 
                                    self.epoch_entry.get(), 
                                    self.bs_entry.get(), 
                                    self.lr_entry.get(), 
                                    self.checkpoint_entry.get(),
                                    evaluation_option,
                                    self.repeat_entry.get())
        except Exception as e:
            raise ValidateException(window=self, message=str(e))

        self.script_history = Script()
        self.script_history.add_import("from XBrainLab.training import TrainingOption")
        self.script_history.add_import("from XBrainLab.training import TRAINING_EVALUATION")
        self.script_history.add_import("import torch")

        self.script_history.add_cmd(f"output_dir={repr(self.training_option.output_dir)}")
        self.script_history.add_cmd(f"optim=torch.optim.{self.optim.__name__}")
        self.script_history.add_cmd(f"optim_parms={repr(self.training_option.optim_parms)}")
        self.script_history.add_cmd(f"use_cpu={repr(self.training_option.use_cpu)}")
        self.script_history.add_cmd(f"gpu_idx={repr(self.training_option.gpu_idx)}")
        self.script_history.add_cmd(f"epoch={repr(self.training_option.epoch)}")
        self.script_history.add_cmd(f"bs={repr(self.training_option.bs)}")
        self.script_history.add_cmd(f"lr={repr(self.training_option.lr)}")
        self.script_history.add_cmd(f"checkpoint_epoch={repr(self.training_option.checkpoint_epoch)}")
        self.script_history.add_cmd(f"evaluation_option={self.training_option.get_evaluation_option_repr()}")
        self.script_history.add_cmd(f"repeat_num={repr(self.training_option.repeat_num)}")
        
        self.script_history.add_cmd(f"training_option = TrainingOption(output_dir=output_dir, ")
        self.script_history.add_cmd(f"optim=optim, optim_parms=optim_parms, ")
        self.script_history.add_cmd(f"use_cpu=use_cpu, gpu_idx=gpu_idx, ")
        self.script_history.add_cmd(f"epoch=epoch, ")
        self.script_history.add_cmd(f"bs=bs, ")
        self.script_history.add_cmd(f"lr=lr, ")
        self.script_history.add_cmd(f"checkpoint_epoch=checkpoint_epoch, ")
        self.script_history.add_cmd(f"evaluation_option=evaluation_option, ")
        self.script_history.add_cmd(f"repeat_num=repeat_num)")
        
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
        self.optim_parms = None

        algo_label = tk.Label(self, text='Algorithm')
        
        algo_map = {c[0]:c[1] for c in inspect.getmembers(torch.optim, inspect.isclass)}
        selected_algo_name = tk.StringVar(self)
        selected_algo_name.trace('w', self.on_algo_select)
        alg_opt = tk.OptionMenu(self, selected_algo_name, *algo_map)
        
        parms_frame = tk.LabelFrame(self, text='Parameters')
        columns = ('parm', 'value')
        parms_tree = EditableTreeView(parms_frame, editableCols=['#2'], columns=columns, show='headings')
        parms_tree.pack(fill=tk.BOTH, expand=True)
        
        PADDING = 5

        algo_label.grid(row=0, column=0, sticky='E', padx=PADDING)
        alg_opt.grid(row=0, column=1, sticky='W', padx=PADDING)
        parms_frame.grid(row=1, column=0, columnspan=2, sticky='NEWS')
        tk.Button(self, text='Confirm', command=self.confirm).grid(row=2, column=0, columnspan=2, pady=PADDING)


        self.algo_map = algo_map 
        self.selected_algo_name = selected_algo_name
        self.parms_tree = parms_tree
        
        selected_algo_name.set(list(algo_map.keys())[0])
        if False: # test code
            selected_algo_name.set('Adam') # test code

    def on_algo_select(self, *args):
        target = self.algo_map[self.selected_algo_name.get()]
        self.parms_tree.delete(*self.parms_tree.get_children())
        if target:
            sigs = inspect.signature(target.__init__)
            parms = list(sigs.parameters)[2:] # skip self and lr
            for parm in parms:
                if 'lr' in parm:
                    continue
                if sigs.parameters[parm].default == inspect._empty or sigs.parameters[parm].default is None:
                    value = ''
                else:
                    value = sigs.parameters[parm].default
                    if False: # test code
                        if parm == 'weight_decay':
                            value = 0.1
                self.parms_tree.insert('', index='end', values=(parm, value))

    def confirm(self):
        self.parms_tree.submitEntry()
        optim_parms = {}
        target = self.algo_map[self.selected_algo_name.get()]
        reason = None
        try:
            for item in self.parms_tree.get_children():
                parm, value = self.parms_tree.item(item)['values']
                if value != '':
                    if isinstance(value, str) and len(value.split()) > 1:
                        value = [float(v) for v in value.split()]
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    else:
                        value = float(value)
                    optim_parms[parm] = value
            target([torch.Tensor()], lr=1, **optim_parms)
        except:
            reason = 'Invalid parameter'
        
        if reason:
            raise ValidateException(window=self, message=reason)
        
        self.optim_parms = optim_parms
        self.optim = target
        
        self.destroy()
    
    def _get_result(self):
        return self.optim, self.optim_parms
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
        result = list(self.device_list.curselection())[0]
        self.use_cpu = 0 == result
        if result > 0:
            self.gpu_idx = result - 1
        self.destroy()
    
    def _get_result(self):
        return self.use_cpu, self.gpu_idx
