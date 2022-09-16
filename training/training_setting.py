import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import inspect

from ..base import TopWindow
from ..widget import EditableTreeView
from .. import model_base
import torch

class TRAINING_EVALUATION(Enum):
    VAL_LOSS = 'Best validation loss'
    TEST_ACC = 'Best testing performance'
    LAST_EPOCH = 'Last Epoch'

def parse_device_name(use_cpu, gpu_idx):
    if use_cpu:
        return 'cpu'
    if gpu_idx is not None:
        return f'{gpu_idx} - {torch.cuda.get_device_name(gpu_idx)}'
    return ''

class TrainingOption:
    def __init__(self, output_dir, optim, optim_parms, use_cpu, gpu_idx, epoch, bs, lr, checkpoint_epoch, evaluation_option, repeat_num):
        self.output_dir = output_dir
        self.optim = optim
        self.optim_parms = optim_parms
        self.use_cpu = use_cpu
        self.gpu_idx = gpu_idx
        self.epoch = epoch
        self.bs = bs
        self.lr = lr
        self.checkpoint_epoch = checkpoint_epoch
        self.evaluation_option = evaluation_option
        self.repeat_num = repeat_num
    
    def get_optim(self, model):
        return self.optim(params=model.parameters(), lr=self.lr, **self.optim_parms)

    def get_optim_name(self):
        return self.optim.__name__

    def get_device_name(self):
        return parse_device_name(self.use_cpu, self.gpu_idx)
    
    def get_device(self):
        if self.use_cpu:
            return 'cpu'
        return f"cuda:{self.gpu_idx}"

class TrainingSettingWindow(TopWindow):
    def __init__(self, parent):
        super().__init__(parent, 'Training Setting')
        self.training_option = None
        self.output_dir = None
        self.optim = None
        self.optim_parms = None
        self.use_cpu = None
        self.gpu_idx = None

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

        epoch_entry.grid(row=0, column=1)
        bs_entry.grid(row=1, column=1)
        lr_entry.grid(row=2, column=1)
        opt_label.grid(row=3, column=1, sticky='EW')
        dev_label.grid(row=4, column=1, sticky='EW')
        output_dir_label.grid(row=5, column=1, sticky='EW')
        checkpoint_entry.grid(row=6, column=1)
        evaluation_option.grid(row=7, column=1)
        repeat_entry.grid(row=8, column=1)
        tk.Button(self, text='Confirm', command=self.confirm).grid(row=9, column=0, columnspan=3)

        self.output_dir_label = output_dir_label
        self.epoch_entry = epoch_entry
        self.bs_entry = bs_entry
        self.lr_entry = lr_entry
        self.checkpoint_entry = checkpoint_entry
        self.dev_label = dev_label
        self.opt_label = opt_label
        self.evaluation_var = evaluation_var
        self.repeat_entry = repeat_entry

        if False: # test code
            epoch_entry.insert(0, '1000')
            bs_entry.insert(0, '288')
            lr_entry.insert(0, '1e-3')
            self.set_optimizer()
            self.set_device()
            self.set_output_dir()
            checkpoint_entry.insert(0, '500') # test code
    
    def set_optimizer(self):
        setter = SetOptimizerWindow(self)
        optim, optim_parms = setter.get_result()
        if not self.winfo_exists():
            return
        if optim and optim_parms:
            self.optim = optim
            self.optim_parms = optim_parms
            self.opt_label.config(text=self.optim.__name__)
        
    def set_device(self):
        setter = SetDeviceWindow(self)
        self.use_cpu, self.gpu_idx = setter.get_result()
        if not self.winfo_exists():
            return
        self.dev_label.config(text=parse_device_name(self.use_cpu, self.gpu_idx))
            
    def set_output_dir(self):
        filepath = tk.filedialog.askdirectory(parent=self)
        if filepath:
            self.output_dir = filepath
            self.output_dir_label.config(text=filepath)

    def confirm(self):
        reason = None
        if self.output_dir is None:
            reason = 'Output directory not set'
        if self.optim  is None or self.optim_parms is None:
            reason = 'Optimizer not set'
        if self.use_cpu is None and self.gpu_idx is None:
            reason = 'Device not set'
        def check_num(i):
            try:
                float(i)
                return False
            except:
                return True

        if check_num(self.epoch_entry.get()):
            reason = 'Invalid epoch'
        if check_num(self.bs_entry.get()):
            reason = 'Invalid batch size'
        if check_num(self.lr_entry.get()):
            reason = 'Invalid learning rate'
        if check_num(self.checkpoint_entry.get()):
            reason = 'Invalid checkpoint epoch'
        if check_num(self.repeat_entry.get()) or int(self.repeat_entry.get()) <= 0:
            reason = 'Invalid repeat number'
        
        evaluation_option = None
        for i in TRAINING_EVALUATION:
            if i.value == self.evaluation_var.get():
                evaluation_option = i
        
        if reason:
            tk.messagebox.showerror('Error',  f"{reason}", parent=self)
            return None
        
        self.training_option = TrainingOption(self.output_dir, self.optim, self.optim_parms, 
                                self.use_cpu, self.gpu_idx, 
                                int(self.epoch_entry.get()), 
                                int(self.bs_entry.get()), 
                                float(self.lr_entry.get()), 
                                int(self.checkpoint_entry.get()),
                                evaluation_option,
                                int(self.repeat_entry.get()))
        self.destroy()
    
    def _get_result(self):
        return self.training_option
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
            tk.messagebox.showerror("Error", reason, parent=self)
            return
        
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
#

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    window = TrainingSettingWindow(root)

    print (window.get_result())
    root.destroy()