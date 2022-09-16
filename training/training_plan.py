import torch
from .trainer import Trainer

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