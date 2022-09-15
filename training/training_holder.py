import torch
from .trainer import Trainer

class ModelHolder:
    def __init__(self, target_model, model_parms_map, pretrained_weight_path):
        self.target_model = target_model
        self.model_parms_map = model_parms_map
        self.pretrained_weight_path = pretrained_weight_path

    def get_model(self, args):
        model = self.target_model(**self.model_parms_map, **args)
        if self.pretrained_weight_path:
            model.load_state_dict(torch.load(self.pretrained_weight_path))
        return model

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

class TrainingPlan:
    def __init__(self, option, model_holder, dataset):
        self.option = option
        self.model_holder = model_holder
        self.dataset = dataset
        self.trainer = None
        self.job = None
        self.error = None

    # get info
    def get_name(self):
        return self.dataset.name
    
    def get_training_status(self):
        if self.trainer is None:
            if self.job:
                return 'Initializing'
            else:
                return 'Not Initialized'
        if self.error:
            return self.error
        if self.trainer.is_finished():
            return 'Finished'
        if not self.job:
            return 'Pending'
        else:
            return f'Training repeat {self.trainer.get_training_repeat()}'

    def get_training_epoch(self):
        if self.trainer is None:
            return ''
        return self.trainer.get_training_epoch()

    def get_training_evaluation(self):
        if self.trainer is None:
            return []
        return self.trainer.get_training_evaluation()

    # interact
    def clear_interrupt(self):
        if self.trainer:
            self.trainer.clear_interrupt()
        self.error = None
    
    def set_interrupt(self):
        if self.trainer:
            self.trainer.set_interrupt()

    def generate_trainer(self):
        self.trainer = Trainer(self.model_holder, self.dataset, self.option)

    def train(self, job):
        self.job = job
        try:
            if self.trainer is None:
                self.generate_trainer()
            self.trainer.train()
            self.job = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = str(e)
            self.job = None

    def get_plans(self):
        if self.trainer is None:
            return []
        return self.trainer.get_plans()