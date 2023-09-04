from __future__ import annotations
import os
import shutil
import torch
import time
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from copy import deepcopy
from ...utils import set_random_state, get_random_state
from ...dataset import Dataset
from ...training import TrainingOption
from .eval import EvalRecord, calculate_confusion

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
        best_val_acc_model: :class:`torch.nn.Module` | None
            Model with best validation accuracy, set during training
        best_val_auc_model: :class:`torch.nn.Module` | None
            Model with best validation auc, set during training
        best_test_acc_model: :class:`torch.nn.Module` | None
            Model with best test accuracy, set during training
        best_test_auc_model: :class:`torch.nn.Module` | None
            Model with best test auc, set during training
        train: dict
            Stores the statistics of each epoch, including loss, accuracy, auc, time used and learning rate
        val: dict
            Stores the statistics of each epoch, including loss, auc and accuracy
        test: dict
            Stores the statistics of each epoch, including loss, auc and accuracy
        best_record: dict
            Stores the statistics of the best model, including best validation loss, best validation auc, best validation accuracy, best test auc, best test accuracy, and their corresponding epoch
        epoch: int
            Current epoch
        target_path: str
            Path to save the record
        random_state: tuple
            Random state for reproducibility
    """
    def __init__(self, repeat: int, dataset: Dataset, model: torch.nn.Module, option: TrainingOption, seed: int):
        self.repeat = repeat
        self.dataset = dataset
        self.option = option
        self.seed = seed
        self.model = model
        self.optim = self.option.get_optim(model)
        self.criterion = self.option.criterion
        #
        self.eval_record = None
        self.best_val_loss_model = None
        self.best_val_acc_model = None
        self.best_val_auc_model = None
        self.best_test_acc_model = None
        self.best_test_auc_model = None
        # 
        self.train = {'loss': [], 'acc': [], 'auc':[], 'time': [], 'lr': []}
        self.val = {'loss': [], 'acc': [], 'auc':[]}
        self.test = {'loss': [], 'acc': [], 'auc':[]}
        self.best_record = {'best_val_loss': torch.inf, 'best_val_acc': -1, 'best_val_auc':-1,
                            'best_test_acc': -1, 'best_test_auc':-1, 
                            'best_val_loss_epoch': None, 'best_val_acc_epoch': None,'best_val_auc_epoch': None, 
                            'best_test_acc_epoch': None, 'best_test_auc_epoch':None}
        #
        self.epoch = 0
        self.target_path = None
        self.create_dir()
        self.random_state = get_random_state()

    def create_dir(self) -> None:
        """Initialize the directory to save the record"""
        record_name = self.dataset.get_name()
        repeat_name = self.get_name()
        target_path = os.path.join(self.option.get_output_dir(), record_name, repeat_name)
        
        if os.path.exists(target_path):
            backup_root = os.path.join(self.option.get_output_dir(), record_name, 'backup')
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

    def update_eval(self, val_acc: float, val_auc: float, val_loss: float) -> None:
        """Append the validation statistics of the current epoch and update the best model"""
        self.append_record(val_acc, self.val['acc'])
        self.append_record(val_auc, self.val['auc'])
        self.append_record(val_loss, self.val['loss'])
        if val_loss <= self.best_record['best_val_loss']:
            self.best_record['best_val_loss'] = val_loss
            self.best_record['best_val_loss_epoch'] = self.epoch + 1
            self.best_val_loss_model = deepcopy(self.model.state_dict())

        if val_acc >= self.best_record['best_val_acc']:
            self.best_record['best_val_acc'] = val_acc
            self.best_record['best_val_acc_epoch'] = self.epoch + 1
            self.best_val_acc_model = deepcopy(self.model.state_dict())

        if val_auc >= self.best_record['best_val_auc']:
            self.best_record['best_val_auc'] = val_auc
            self.best_record['best_val_auc_epoch'] = self.epoch + 1
            self.best_val_auc_model = deepcopy(self.model.state_dict())
    
    def update_test(self, test_acc: float, test_auc: float, test_loss: float) -> None:
        """Append the test statistics of the current epoch and update the best model"""
        self.append_record(test_acc, self.test['acc'])
        self.append_record(test_auc, self.test['auc'])
        self.append_record(test_loss, self.test['loss'])
        if test_acc >= self.best_record['best_test_acc']:
            self.best_record['best_test_acc'] = test_acc
            self.best_record['best_test_acc_epoch'] = self.epoch + 1
            self.best_test_acc_model = deepcopy(self.model.state_dict())
        if test_auc >= self.best_record['best_test_auc']:
            self.best_record['best_test_auc'] = test_auc
            self.best_record['best_test_auc_epoch'] = self.epoch + 1
            self.best_test_auc_model = deepcopy(self.model.state_dict())

    def update_train(self, train_acc: float, train_auc: float, running_loss: float) -> None:
        """Append the training statistics of the current epoch"""
        self.append_record(train_acc, self.train['acc'])
        self.append_record(train_auc, self.train['auc'])
        self.append_record(running_loss, self.train['loss'])
    
    def step(self, trainingTime: float, lr: float) -> None:
        """Append the time and learning rate of the current epoch and move to the next epoch"""
        self.append_record(trainingTime, self.train['time'])
        self.append_record(lr, self.train['lr'])
        self.epoch += 1

    def set_eval_record(self, eval_record: EvalRecord) -> None:
        """Set the evaluation record when training is finished"""
        self.eval_record = eval_record
    # 
    def export_checkpoint(self) -> None:
        """Export the checkpoint of the training record"""
        epoch = len(self.train['loss'])
        if self.eval_record:
            self.eval_record.export(self.target_path)
        if self.best_val_loss_model:
            torch.save(self.best_val_loss_model, os.path.join(self.target_path, 'best_val_loss_model'))
        if self.best_val_acc_model:
            torch.save(self.best_val_acc_model, os.path.join(self.target_path, 'best_val_acc_model'))
        if self.best_val_auc_model:
            torch.save(self.best_val_auc_model, os.path.join(self.target_path, 'best_val_auc_model'))
        if self.best_test_acc_model:
            torch.save(self.best_test_acc_model, os.path.join(self.target_path, 'best_test_acc_model'))
        if self.best_test_auc_model:
            torch.save(self.best_test_auc_model, os.path.join(self.target_path, 'best_test_auc_model'))
        
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
    def get_loss_figure(self, fig: Figure = None, figsize: tuple = (6.4, 4.8), dpi: int = 100) -> Figure:
        """Return the line chart of loss during training
        
        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        training_loss_list = self.train['loss']
        val_loss_list = self.val['loss']
        test_loss_list = self.test['loss']
        if len(training_loss_list) == 0 and len(val_loss_list) == 0 and len(test_loss_list) == 0:
            return None

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

    def get_acc_figure(self, fig: Figure = None, figsize: tuple = (6.4, 4.8), dpi: int = 100) -> Figure:
        """Return the line chart of accuracy during training

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        training_acc_list = self.train['acc']
        val_acc_list = self.val['acc']
        test_acc_list = self.test['acc']
        if len(training_acc_list) == 0 and len(val_acc_list) == 0 and len(test_acc_list) == 0:
            return None
        plt.plot(training_acc_list, 'g', label='Training accuracy')
        if len(val_acc_list) > 0:
            plt.plot(val_acc_list, 'b', label='validation accuracy')
        if len(val_acc_list) > 0:
            plt.plot(test_acc_list, 'r', label='testing accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        _ = plt.legend(loc='upper left')
        
        return fig

    def get_auc_figure(self, fig: Figure = None, figsize: tuple = (6.4, 4.8), dpi: int = 100) -> Figure:
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
        
        training_auc_list = self.train['auc']
        val_auc_list = self.val['auc']
        test_auc_list = self.test['auc']
        if len(training_auc_list) == 0 and len(val_auc_list) == 0 and len(test_auc_list) == 0:
            return None
        plt.plot(training_auc_list, 'g', label='Training AUC')
        if len(val_auc_list) > 0:
            plt.plot(val_auc_list, 'b', label='validation AUC')
        if len(val_auc_list) > 0:
            plt.plot(test_auc_list, 'r', label='testing AUC')
        plt.title('Training AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        _ = plt.legend(loc='upper left')
        
        return fig
    
    def get_lr_figure(self, fig: Figure = None, figsize: tuple = (6.4, 4.8), dpi: int = 100) -> Figure:
        """Return the line chart of learning rate during training

        Args:
            fig: Figure to be plotted on. If None, a new figure will be created
            figsize: Figure size
            dpi: Figure dpi
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        lr_list = self.train['lr']
        if len(lr_list) == 0:
            return None

        plt.plot(lr_list, 'g')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('lr')
        return fig

    def get_confusion_figure(self, fig: Figure = None, figsize: tuple = (6.4, 4.8), dpi: int = 100) -> Figure:
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
        ax.set_title(f'Confusion matrix')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        res = ax.imshow(confusion, cmap='magma', interpolation='nearest')
        for x in range(classNum):
            for y in range(classNum):
                annot_color = 'k' if confusion[x][y]>(confusion.max()-confusion.min())/2 else 'w'
                ax.annotate(str(confusion[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center',
                            color=annot_color
                            )
        cb = fig.colorbar(res)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        labels = [self.dataset.get_epoch_data().label_map[l] for l in range(classNum)]
        plt.xticks(range(classNum), labels)
        plt.yticks(range(classNum), labels)

        return fig
    
    # get evaluate
    def get_acc(self) -> float | None:
        """Get the accuracy of the evaluation record, None if training is not finished"""
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
