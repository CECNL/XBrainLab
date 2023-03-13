from enum import Enum
import torch
import torch.nn as nn

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

def parse_optim_name(optim, optim_parms):
    option_list = [f"{i}={optim_parms[i]}" for i in optim_parms if optim_parms[i] ]
    options = ', '.join(option_list)
    return f"{optim.__name__} ({options})"

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
        self.criterion = nn.CrossEntropyLoss()
        self.validate()

    def validate(self):
        reason = None
        if self.output_dir is None:
            reason = 'Output directory not set'
        if self.optim  is None or self.optim_parms is None:
            reason = 'Optimizer not set'
        if self.use_cpu is None:
            reason = 'Device not set'
        if not self.use_cpu and self.gpu_idx is None:
            reason = 'Device not set'
        if self.evaluation_option is None:
            reason = 'Evaluation option not set'

        def check_num(i):
            try:
                float(i)
                return False
            except:
                return True

        if self.gpu_idx is not None:
            if check_num(self.gpu_idx):
                reason = 'Invalid gpu_idx'
        if check_num(self.epoch):
            reason = 'Invalid epoch'
        if check_num(self.bs):
            reason = 'Invalid batch size'
        if check_num(self.lr):
            reason = 'Invalid learning rate'
        if check_num(self.checkpoint_epoch):
            reason = 'Invalid checkpoint epoch'
        if check_num(self.repeat_num) or int(self.repeat_num) <= 0:
            reason = 'Invalid repeat number'
        
        if reason:
            raise ValueError(reason)
        
        self.epoch = int(self.epoch)
        self.bs = int(self.bs) 
        self.lr = float(self.lr) 
        self.checkpoint_epoch = int(self.checkpoint_epoch)
        self.repeat_num = int(self.repeat_num)
        if self.gpu_idx is not None:
            self.gpu_idx = int(self.gpu_idx)
    
    def get_optim(self, model):
        return self.optim(params=model.parameters(), lr=self.lr, **self.optim_parms)

    def get_optim_name(self):
        return self.optim.__name__

    def get_optim_desc_str(self):
        return parse_optim_name(self.optim, self.optim_parms)

    def get_device_name(self):
        return parse_device_name(self.use_cpu, self.gpu_idx)
    
    def get_device(self):
        if self.use_cpu:
            return 'cpu'
        return f"cuda:{self.gpu_idx}"

    def get_evaluation_option_repr(self):
        return f"{self.evaluation_option.__class__.__name__}.{self.evaluation_option.name}"

    def get_output_dir(self):
        return self.output_dir


class TestOnlyOption(TrainingOption):
    def __init__(self, output_dir, use_cpu, gpu_idx, bs):
        super().__init__(output_dir, None, None, use_cpu, gpu_idx, 0, bs, 0, 0, TRAINING_EVALUATION.LAST_EPOCH, 1)
        self.validate()

    def validate(self):
        reason = None
        if self.output_dir is None:
            reason = 'Output directory not set'
        if self.use_cpu is None:
            reason = 'Device not set'
        if not self.use_cpu and self.gpu_idx is None:
            reason = 'Device not set'
        if self.evaluation_option is None:
            reason = 'Evaluation option not set'

        def check_num(i):
            try:
                float(i)
                return False
            except:
                return True

        if self.gpu_idx is not None:
            if check_num(self.gpu_idx):
                reason = 'Invalid gpu_idx'
        if check_num(self.epoch):
            reason = 'Invalid epoch'
        if check_num(self.bs):
            reason = 'Invalid batch size'
        
        if reason:
            raise ValueError(reason)
        
        self.epoch = int(self.epoch)
        self.bs = int(self.bs) 
        self.repeat_num = int(self.repeat_num)
        if self.gpu_idx is not None:
            self.gpu_idx = int(self.gpu_idx)
    
    def get_optim(self, model):
        return None

    def get_optim_name(self):
        return '-'

    def get_optim_desc_str(self):
        return '-'

    def get_device_name(self):
        return parse_device_name(self.use_cpu, self.gpu_idx)
    
    def get_device(self):
        if self.use_cpu:
            return 'cpu'
        return f"cuda:{self.gpu_idx}"

    def get_evaluation_option_repr(self):
        return f"{self.evaluation_option.__class__.__name__}.{self.evaluation_option.name}"

    def get_output_dir(self):
        return self.output_dir