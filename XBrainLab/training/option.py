from enum import Enum

import torch
from torch import nn


class TRAINING_EVALUATION(Enum):
    """Utility class for model selection option"""
    VAL_LOSS = 'Best validation loss'
    TEST_AUC = 'Best testing AUC'
    TEST_ACC = 'Best testing performance'
    LAST_EPOCH = 'Last Epoch'

def parse_device_name(use_cpu: bool, gpu_idx: int) -> str:
    """Return device description string"""
    if use_cpu:
        return 'cpu'
    if gpu_idx is not None:
        return f'{gpu_idx} - {torch.cuda.get_device_name(gpu_idx)}'
    raise ValueError('Device not set')

def parse_optim_name(optim: type, optim_params: dict) -> str:
    """Return optimizer description string, including optimizer name and parameters"""
    option_list = [f"{i}={optim_params[i]}" for i in optim_params if optim_params[i]]
    options = ', '.join(option_list)
    return f"{optim.__name__} ({options})"

class TrainingOption:
    """Utility class for storing training options

    Attributes:
        output_dir: Output directory
        optim: Optimizer class of type :class:`torch.optim.Optimizer`
        optim_params: Optimizer parameters
        use_cpu: Whether to use CPU
        gpu_idx: GPU index
        epoch: Number of epochs
        bs: Batch size
        lr: Learning rate
        checkpoint_epoch: Checkpoint epoch
        evaluation_option: Model selection option
        repeat_num: Number of repeats
        criterion: Loss function
    """
    def __init__(self,
                 output_dir: str,
                 optim: type,
                 optim_params: dict,
                 use_cpu: bool,
                 gpu_idx: int,
                 epoch: int,
                 bs: int,
                 lr: float,
                 checkpoint_epoch: int,
                 evaluation_option: TRAINING_EVALUATION,
                 repeat_num: int):
        self.output_dir = output_dir
        self.optim = optim
        self.optim_params = optim_params
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

    def validate(self) -> None:
        """Validate training options

        Raises:
            ValueError: If any option is invalid or not set
        """
        reason = None
        if self.output_dir is None:
            reason = 'Output directory not set'
        if self.optim  is None or self.optim_params is None:
            reason = 'Optimizer not set'
        if self.use_cpu is None:
            reason = 'Device not set'
        if not self.use_cpu and self.gpu_idx is None:
            reason = 'Device not set'
        if self.evaluation_option is None:
            reason = 'Evaluation option not set'

        def check_num(i):
            """Return True if i is not a number"""
            try:
                float(i)
            except Exception:
                return True
            else:
                return False


        if self.gpu_idx is not None and check_num(self.gpu_idx):
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

    def get_optim(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Return optimizer instance"""
        return self.optim(params=model.parameters(), lr=self.lr, **self.optim_params)

    def get_optim_name(self) -> str:
        """Return optimizer name"""
        return self.optim.__name__

    def get_optim_desc_str(self) -> str:
        """Return optimizer description string,
           including optimizer name and parameters"""
        return parse_optim_name(self.optim, self.optim_params)

    def get_device_name(self) -> str:
        """Return device description string"""
        return parse_device_name(self.use_cpu, self.gpu_idx)

    def get_device(self) -> str:
        """Return device name used by PyTorch"""
        if self.use_cpu:
            return 'cpu'
        return f"cuda:{self.gpu_idx}"

    def get_evaluation_option_repr(self) -> str:
        """Return model selection option description string"""
        module_name = self.evaluation_option.__class__.__name__
        class_name = self.evaluation_option.name
        return f"{module_name}.{class_name}"

    def get_output_dir(self) -> str:
        """Return output directory"""
        return self.output_dir


class TestOnlyOption(TrainingOption):
    __test__ = False # Not a test case
    """Utility class for storing test-only options

    Parameters:
        output_dir: Output directory
        use_cpu: Whether to use CPU
        gpu_idx: GPU index
        bs: Batch size
    """
    def __init__(self, output_dir: str, use_cpu: bool, gpu_idx: int, bs: int):
        super().__init__(
            output_dir, None, None, use_cpu, gpu_idx, 0, bs, 0, 0,
            TRAINING_EVALUATION.LAST_EPOCH, 1
        )
        self.validate()

    def validate(self) -> None:
        """Validate test-only options

        Raises:
            ValueError: If any option is invalid or not set
        """
        reason = None
        if self.output_dir is None:
            reason = 'Output directory not set'
        if self.use_cpu is None:
            reason = 'Device not set'
        if not self.use_cpu and self.gpu_idx is None:
            reason = 'Device not set'

        def check_num(i):
            """Return True if i is not a number"""
            try:
                float(i)
            except Exception:
                return True
            else:
                return False

        if self.gpu_idx is not None and check_num(self.gpu_idx):
            reason = 'Invalid gpu_idx'
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
        module_name = self.evaluation_option.__class__.__name__
        class_name = self.evaluation_option.name
        return f"{module_name}.{class_name}"

    def get_output_dir(self):
        return self.output_dir
