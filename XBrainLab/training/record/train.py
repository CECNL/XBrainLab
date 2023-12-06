from __future__ import annotations

import os
import shutil
import time
from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ...dataset import Dataset
from ...training import TrainingOption
from ...utils import get_random_state, set_random_state
from .eval import EvalRecord, calculate_confusion


class RecordKey:
    "Utility class for accessing the statistics of the testing record"
    LOSS = 'loss'
    ACC = 'accuracy'
    AUC = 'auc'

    def __iter__(self):
        keys = dir(self)
        keys = [getattr(self, key) for key in keys if not key.startswith('_')]
        return iter(keys)

class TrainRecordKey(RecordKey):
    "Utility class for accessing the statistics of the training record"
    TIME = 'time'
    LR = 'lr'

class TrainRecord:
    """Class for recording statistics during training

    Attributes:
        repeat: int
            Index of the repeat
        dataset: :class:`XBrainLab.dataset.Dataset`
            Dataset used for training
        model: :class:`torch.nn.Module`
            Model used for training
        option: :class:`XBrainLab.training.TrainingOption`
            Training option
        seed: int
            Random seed
        optim: :class:`torch.optim.Optimizer`
            Optimizer used for training
        criterion: :class:`torch.nn.Module`
            Criterion used for training
        eval_record: :class:`XBrainLab.training.record.EvalRecord` | None
            Evaluation record, set after training is finished
        best_val_loss_model: :class:`torch.nn.Module` | None
            Model with best validation loss, set during training
        best_val_accuracy_model: :class:`torch.nn.Module` | None
            Model with best validation accuracy, set during training
        best_val_auc_model: :class:`torch.nn.Module` | None
            Model with best validation auc, set during training
        best_test_accuracy_model: :class:`torch.nn.Module` | None
            Model with best test accuracy, set during training
        best_test_auc_model: :class:`torch.nn.Module` | None
            Model with best test auc, set during training
        train: dict
            Stores the statistics of each epoch, including loss, accuracy, auc,
            time used and learning rate
        val: dict
            Stores the statistics of each epoch, including loss, auc and accuracy
        test: dict
            Stores the statistics of each epoch, including loss, auc and accuracy
        best_record: dict
            Stores the statistics of the best model, including best validation loss,
            best validation auc, best validation accuracy, best test auc,
            best test accuracy, and their corresponding epoch
        epoch: int
            Current epoch
        target_path: str
            Path to save the record
        random_state: tuple
            Random state for reproducibility
    """
    def __init__(
        self,
        repeat: int,
        dataset: Dataset,
        model: torch.nn.Module,
        option: TrainingOption,
        seed: int
    ):
        self.repeat = repeat
        self.dataset = dataset
        self.option = option
        self.seed = seed
        self.model = model
        self.optim = self.option.get_optim(model)
        self.criterion = self.option.criterion
        #
        self.eval_record = None
        for key in RecordKey():
            setattr(self, 'best_val_' + key + '_model', None)
            setattr(self, 'best_test_' + key + '_model', None)
        #
        self.train = {i: [] for i in TrainRecordKey()}
        self.val = {i: [] for i in RecordKey()}
        self.test = {i: [] for i in RecordKey()}
        self.best_record = {}
        for record_type in ['val', 'test']:
            for key in RecordKey():
                self.best_record[f'best_{record_type}_{key}'] = -1
                self.best_record[f'best_{record_type}_{key}_epoch'] = None
            self.best_record[f'best_{record_type}_' + RecordKey.LOSS] = torch.inf

        #
        self.epoch = 0
        self.target_path = None
        self.create_dir()
        self.random_state = get_random_state()

    def create_dir(self) -> None:
        """Initialize the directory to save the record"""
        record_name = self.dataset.get_name()
        repeat_name = self.get_name()
        target_path = os.path.join(
            self.option.get_output_dir(), record_name, repeat_name
        )

        if os.path.exists(target_path):
            backup_root = os.path.join(
                self.option.get_output_dir(), record_name, 'backup'
            )
            backup_path = os.path.join(backup_root, f"{repeat_name}-{time.time()}")
            os.makedirs(backup_root, exist_ok=True)
            shutil.move(target_path, backup_path)

        os.makedirs(target_path)
        self.target_path = target_path

    def resume(self) -> None:
        """Resume training from the last training state"""
        set_random_state(self.random_state)

    def pause(self) -> None:
        """Pause training and save the current training state"""
        self.random_state = get_random_state()

    def get_name(self) -> str:
        """Return the name of the record"""
        return f"Repeat-{self.repeat}"

    def get_epoch(self) -> int:
        """Get the current epoch"""
        return self.epoch

    def get_training_model(self, device: str) -> torch.nn.Module:
        """Get the model for training and move it to the device"""
        return self.model.to(device)

    def is_finished(self) -> bool:
        """Check if the training is finished"""
        return self.get_epoch() >= self.option.epoch and self.eval_record is not None
    #
    def append_record(self, val: any, arr: list) -> None:
        """Internal function for appending a value to a statistic array

        Fill the array with None if the data is not available before the current epoch

        Args:
            val: Value to be appended
            arr: Array to be appended
        """
        while len(arr) < self.epoch:
            arr.append(None)
        if len(arr) > self.epoch:
            arr[self.epoch] = val
        elif len(arr) == self.epoch:
            arr.append(val)

    def update(self, update_type: str, test_result: dict[str, float]) -> None:
        """Append the statistics of given type the current epoch"""
        for key in test_result:
            self.append_record(test_result[key], getattr(self, update_type)[key])
            should_update = False
            if 'loss' in key:
                if (
                    test_result[key] <=
                    self.best_record['best_' + update_type + '_' + key]
                ):
                    should_update = True
            elif (
                    test_result[key] >=
                    self.best_record['best_' + update_type + '_' + key]
                ):
                    should_update = True
            if should_update:
                self.best_record['best_' + update_type + '_' + key] = test_result[key]
                self.best_record['best_' + update_type + '_' + key + '_epoch'] = \
                    self.get_epoch()
                setattr(
                    self, 'best_' + update_type + '_' + key + '_model',
                    deepcopy(self.model.state_dict())
                )

    def update_eval(self, test_result: dict[str, float]) -> None:
        """Append the validation statistics of the current epoch and
        update the best model"""
        self.update('val', test_result)

    def update_test(self, test_result: dict[str, float]) -> None:
        """Append the test statistics of the current epoch and update the best model"""
        self.update('test', test_result)

    def update_train(self, test_result: dict[str, float]) -> None:
        """Append the training statistics of the current epoch"""
        for key in test_result:
            self.append_record(test_result[key], self.train[key])

    def update_statistic(self, statistic: dict[str, float]) -> None:
        """Append the statistics of the current epoch"""
        for key in statistic:
            self.append_record(statistic[key], self.train[key])

    def step(self) -> None:
        """Move to the next epoch"""
        self.epoch += 1

    def set_eval_record(self, eval_record: EvalRecord) -> None:
        """Set the evaluation record when training is finished"""
        self.eval_record = eval_record
    #
    def export_checkpoint(self) -> None:
        """Export the checkpoint of the training record"""
        epoch = len(self.train[RecordKey.LOSS])
        if self.eval_record:
            self.eval_record.export(self.target_path)
        for best_type in ['val', 'test']:
            for key in RecordKey():
                full_key = 'best_' + best_type + '_' + key + '_model'
                model = getattr(self, full_key)
                if model:
                    torch.save(model, os.path.join(self.target_path, full_key))

        fname = f'Epoch-{epoch}-model'
        torch.save(self.model.state_dict(), os.path.join(self.target_path, fname))
        #
        record = {
            'train': self.train,
            'val': self.val,
            'test': self.test,
            'best_record': self.best_record,
            'seed': self.seed
        }
        torch.save(record, os.path.join(self.target_path, 'record'))


    # figure
    def get_loss_figure(
        self,
        fig: Figure = None,
        figsize: tuple = (6.4, 4.8),
        dpi: int = 100
    ) -> Figure:
        """Return the line chart of loss during training

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        training_loss_list = self.train[RecordKey.LOSS]
        val_loss_list = self.val[RecordKey.LOSS]
        test_loss_list = self.test[RecordKey.LOSS]
        if (
            len(training_loss_list) == 0 and
            len(val_loss_list) == 0 and
            len(test_loss_list) == 0
        ):
            return None

        if len(training_loss_list) > 0:
            plt.plot(training_loss_list, 'g', label='Training loss')
        if len(val_loss_list) > 0:
            plt.plot(val_loss_list, 'b', label='validation loss')
        if len(test_loss_list) > 0:
            plt.plot(test_loss_list, 'r', label='testing loss')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        _ = plt.legend(loc='center left')

        return fig

    def get_acc_figure(
        self,
        fig: Figure = None,
        figsize: tuple = (6.4, 4.8),
        dpi: int = 100
    ) -> Figure:
        """Return the line chart of accuracy during training

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        training_acc_list = self.train[RecordKey.ACC]
        val_acc_list = self.val[RecordKey.ACC]
        test_acc_list = self.test[RecordKey.ACC]
        if (
            len(training_acc_list) == 0 and
            len(val_acc_list) == 0 and
            len(test_acc_list) == 0
        ):
            return None

        if len(training_acc_list) > 0:
            plt.plot(training_acc_list, 'g', label='Training accuracy')
        if len(val_acc_list) > 0:
            plt.plot(val_acc_list, 'b', label='validation accuracy')
        if len(test_acc_list) > 0:
            plt.plot(test_acc_list, 'r', label='testing accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        _ = plt.legend(loc='upper left')

        return fig

    def get_auc_figure(
        self,
        fig: Figure = None,
        figsize: tuple = (6.4, 4.8),
        dpi: int = 100
    ) -> Figure:
        """Return the line chart of auc during training

        TODO:

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        training_auc_list = self.train[RecordKey.AUC]
        val_auc_list = self.val[RecordKey.AUC]
        test_auc_list = self.test[RecordKey.AUC]
        if (
            len(training_auc_list) == 0 and
            len(val_auc_list) == 0 and
            len(test_auc_list) == 0
        ):
            return None

        if len(training_auc_list) > 0:
            plt.plot(training_auc_list, 'g', label='Training AUC')
        if len(val_auc_list) > 0:
            plt.plot(val_auc_list, 'b', label='validation AUC')
        if len(test_auc_list) > 0:
            plt.plot(test_auc_list, 'r', label='testing AUC')
        plt.title('Training AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        _ = plt.legend(loc='upper left')

        return fig

    def get_lr_figure(
        self,
        fig: Figure = None,
        figsize: tuple = (6.4, 4.8),
        dpi: int = 100
    ) -> Figure:
        """Return the line chart of learning rate during training

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        lr_list = self.train[TrainRecordKey.LR]
        if len(lr_list) == 0:
            return None

        plt.plot(lr_list, 'g')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('lr')
        return fig

    def get_confusion_figure(
        self,
        fig: Figure = None,
        figsize: tuple = (6.4, 4.8),
        dpi: int = 100
    ) -> Figure:
        """Return the confusion matrix of the evaluation record

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        if not self.eval_record:
            return None
        output = self.eval_record.output
        label = self.eval_record.label
        confusion = calculate_confusion(output, label)
        classNum = confusion.shape[0]

        ax = fig.add_subplot(111)
        ax.set_title('Confusion matrix')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        res = ax.imshow(confusion, cmap='magma', interpolation='nearest')
        for x in range(classNum):
            for y in range(classNum):
                if confusion[x][y] > (confusion.max() - confusion.min()) / 2:
                    annot_color = 'k'
                else:
                    annot_color = 'w'
                ax.annotate(str(confusion[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color=annot_color
                            )
        fig.colorbar(res)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        labels = [self.dataset.get_epoch_data().label_map[i] for i in range(classNum)]
        plt.xticks(range(classNum), labels)
        plt.yticks(range(classNum), labels)

        return fig

    # get evaluate
    def get_acc(self) -> float | None:
        """Get the accuracy of the evaluation record,
        None if training is not finished"""
        if not self.eval_record:
            return None
        return self.eval_record.get_acc()

    def get_auc(self) -> float | None:
        """Get the auc of the evaluation record, None if training is not finished"""
        if not self.eval_record:
            return None
        return self.eval_record.get_auc()

    def get_kappa(self) -> float | None:
        """Get the kappa of the evaluation record, None if training is not finished"""
        if not self.eval_record:
            return None
        return self.eval_record.get_kappa()

    def get_eval_record(self) -> EvalRecord | None:
        """Get the evaluation record, None if training is not finished"""
        return self.eval_record
