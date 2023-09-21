import threading
from enum import Enum
from typing import List

from ..utils import validate_list_type
from .training_plan import TrainingPlanHolder


class Status(Enum):
    """Utility class for training status"""
    PENDING = 'Pending'
    INIT = 'Initializing'
    INTING = 'Interrupting'
    TRAIN = 'Now training: {}'

class Trainer:
    """Class for storing training options and training models

    Attributes:
        interrupt: bool
            Whether to interrupt training
        progress_text: :class:`Status`
            Training progress
        training_plan_holders: List[:class:`TrainingPlanHolder`]
            List of training plan holders
        job_thread: :class:`threading.Thread`
            Thread for training in background
    """
    def __init__(self, training_plan_holders: List[TrainingPlanHolder]):
        validate_list_type(
            training_plan_holders, TrainingPlanHolder, 'training_plan_holders'
        )
        self.interrupt = False
        self.progress_text = Status.PENDING
        self.training_plan_holders = training_plan_holders
        self.job_thread = None

    def get_training_plan_holders(self) -> List[TrainingPlanHolder]:
        """Return list of training plan holders"""
        return self.training_plan_holders

    def get_real_training_plan(
            self,
            plan_name: str, real_plan_name: str
        ) -> TrainingPlanHolder:
        """Return training plan by name"""
        for training_plan_holder in self.training_plan_holders:
            if training_plan_holder.get_name() == plan_name:
                for plan in training_plan_holder.get_plans():
                    if plan.get_name() == real_plan_name:
                        return plan
                raise ValueError(
                    f'No real plan named "{real_plan_name}" for "{plan_name}"'
                )
        raise ValueError(f'No training plan named "{plan_name}"')

    def set_interrupt(self) -> None:
        """Set interrupt flag to True and interrupt all training plan holders"""
        self.progress_text = Status.INTING
        self.interrupt = True
        for plan_holder in self.training_plan_holders:
            plan_holder.set_interrupt()

    def clear_interrupt(self) -> None:
        """Reset interrupt flag

        Set interrupt flag to False and
        clear interrupt flag of all training plan holders
        """
        self.progress_text = Status.PENDING
        self.interrupt = False
        for plan_holder in self.training_plan_holders:
            plan_holder.clear_interrupt()

    def job(self) -> None:
        """Training job running in background"""
        for plan_holder in self.training_plan_holders:
            self.progress_text = Status.TRAIN.value.format(plan_holder.get_name())
            if self.interrupt:
                break
            plan_holder.train()
        self.progress_text = Status.PENDING
        self.job_thread = None

    def run(self, interact: bool = False) -> None:
        """Run training job

        Parameters:
            interact: bool
                Whether to run training in background
        """
        if self.is_running():
            return

        self.clear_interrupt()
        if interact:
            self.job_thread = threading.Thread(target=self.job)
            self.job_thread.start()
        else:
            self.job()

    def get_progress_text(self) -> str:
        """Return string representation of training progress"""
        if isinstance(self.progress_text, Status):
            return self.progress_text.value
        return self.progress_text

    def is_running(self) -> bool:
        """Return whether training is running"""
        return self.job_thread is not None and self.job_thread.is_alive()

    def clean(self, force_update: bool = False) -> None:
        """Stop and clean training job

        Parameters:
            force_update: bool
                Whether to force update

        Raises:
            ValueError: If training is still in progress and
                        :attr:`force_update` is False
        """
        if force_update:
            self.set_interrupt()
        elif self.is_running():
            raise RuntimeError("Training still in progress")
