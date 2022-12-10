import threading
from enum import Enum

from ..utils import validate_list_type
from . import TrainingPlanHolder

class Status(Enum):
    PENDING = 'Pending'
    INIT = 'Initializing'
    INTING = 'Interrupting'
    TRAIN = 'Now training: {}'

class Trainer():
    def __init__(self, training_plan_holders):
        validate_list_type(training_plan_holders, TrainingPlanHolder, 'training_plan_holders')
        self.interrupt = False
        self.progress_text = Status.PENDING
        self.training_plan_holders = training_plan_holders
        self.job_thread = None
    
    def get_training_plan_holders(self):
        return self.training_plan_holders

    def set_interrupt(self):
        self.progress_text = Status.INTING
        self.interrupt = True
        for plan_holder in self.training_plan_holders:
            plan_holder.set_interrupt()

    def clear_interrupt(self):
        self.progress_text = Status.INIT
        self.interrupt = False
        for plan_holder in self.training_plan_holders:
            plan_holder.clear_interrupt()

    def job(self):
        for plan_holder in self.training_plan_holders:
            self.progress_text = Status.TRAIN.value.format(plan_holder.get_name())
            if self.interrupt:
                break
            plan_holder.train()
        self.progress_text = Status.PENDING
        self.job_thread = None

    def run(self, interact=False):
        if self.is_running():
            return
        
        self.clear_interrupt()
        if interact:
            self.job_thread = threading.Thread(target=self.job)
            self.job_thread.start()
        else:
            self.job()

    def get_progress_text(self):
        if isinstance(self.progress_text, Status):
            return self.progress_text.value
        return self.progress_text

    def is_running(self):
        return self.job_thread is not None

    def clean(self, force_update=False):
        if force_update:
            self.set_interrupt()
        elif self.is_running():
            raise ValueError("Training still in progress")
