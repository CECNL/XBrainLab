import torch

class ModelHolder:
    def __init__(self, target_model, model_parms_map, pretrained_weight_path):
        self.target_model = target_model
        self.model_parms_map = model_parms_map
        self.pretrained_weight_path = pretrained_weight_path

def parse_device_name(use_cpu, gpu_idx):
    if use_cpu:
        return 'cpu'
    if gpu_idx:
        return f'{gpu_idx} - {torch.cuda.get_device_name(gpu_idx)}'
    return ''

class TrainingOption:
    def __init__(self, output_dir, optim, optim_parms, use_cpu, gpu_idx, epoch, bs, lr, checkpoint_epoch):
        self.output_dir = output_dir
        self.optim = optim
        self.optim_parms = optim_parms
        self.use_cpu = use_cpu
        self.gpu_idx = gpu_idx
        self.epoch = epoch
        self.bs = bs
        self.lr = lr
        self.checkpoint_epoch = checkpoint_epoch
    
    def get_optim(self, model):
        return self.optim(params=model.parameters(), lr=self.lr, **self.optim_parms)

    def get_optim_name(self):
        return self.optim.__name__

    def get_device_name(self):
        return parse_device_name(self.use_cpu, self.gpu_idx)

class TrainingSetting:
    def __init__(self):
        pass