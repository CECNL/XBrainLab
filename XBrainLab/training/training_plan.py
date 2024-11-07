from __future__ import annotations

import time
import traceback
from enum import Enum

import numpy as np
import torch
import torch.utils.data as Data
from captum.attr import Saliency, NoiseTunnel
from sklearn.metrics import roc_auc_score

from ..dataset import Dataset
from ..visualization import supported_saliency_methods
from ..utils import set_seed, validate_type
from .model_holder import ModelHolder
from .option import TRAINING_EVALUATION, TrainingOption
from .record import EvalRecord, RecordKey, TrainRecord, TrainRecordKey


def _test_model(
    model: torch.nn.Module,
    dataLoader: Data.DataLoader,
    criterion: torch.nn.Module
) -> dict[str, float]:
    """Test model on given data loader

    Args:
        model: Model to be tested
        dataLoader: Data loader
        criterion: Loss function

    Returns:
        Dictionary of test result, including loss, accuracy and auc
    """
    model.eval()

    running_loss = 0.0
    total_count = 0
    auc = 0
    correct = 0
    y_true, y_pred = None, None
    with torch.no_grad():
        for inputs, labels in dataLoader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            correct += (outputs.argmax(axis=1) == labels).float().sum().item()
            total_count += len(labels)

            if y_true is None or y_pred is None:
                y_true = labels
                y_pred = outputs
            else:
                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, outputs))

        try:
            if y_pred.size()[-1] <=2:
                auc = roc_auc_score(y_true.clone().detach().cpu().numpy(),
                torch.nn.functional.softmax(
                    y_pred, dim=1
                ).clone().detach().cpu().numpy()[:, 1])
            else:
                auc = roc_auc_score(y_true.clone().detach().cpu().numpy(),
                torch.nn.functional.softmax(
                    y_pred, dim=1
                ).clone().detach().cpu().numpy(), multi_class='ovr')
        except Exception:
            pass

    running_loss /= len(dataLoader)
    acc = correct / total_count * 100
    return {
        RecordKey.ACC: acc,
        RecordKey.AUC: auc,
        RecordKey.LOSS: running_loss
    }

def _eval_model(model: torch.nn.Module, dataLoader: Data.DataLoader, saliency_params:dict) -> EvalRecord:
    """Evaluate model on given data loader

    Evaluate model and compute saliency map for each class

    Args:
        model: Model to be evaluated
        dataLoader: Data loader
    """
    model.eval()

    input_list = []
    output_list = []
    label_list = []

    gradient_list = []
    smoothgrad_list = []
    smoothgrad_sq_list = []
    vargrad_list = []
    saliency_inst = Saliency(model)
    noise_tunnel_inst = NoiseTunnel(saliency_inst)

    for inputs, labels in dataLoader:
        inputs.requires_grad = False
        outputs = model(inputs)

        input_list.append(inputs.detach().cpu().numpy())
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        inputs.requires_grad=True
        gradient_list.append(
            saliency_inst.attribute(
                inputs, target=labels.detach().cpu().numpy().tolist(), abs=False
            ).detach().cpu().numpy()
        )
        smoothgrad_list.append(
            noise_tunnel_inst.attribute(
                inputs, target=labels.detach().cpu().numpy().tolist(),
                nt_type='smoothgrad', **saliency_params['SmoothGrad']
            ).detach().cpu().numpy()
        )
        smoothgrad_sq_list.append(
            noise_tunnel_inst.attribute(
                inputs, target=labels.detach().cpu().numpy().tolist(),
                nt_type='smoothgrad_sq', **saliency_params['SmoothGrad Squared']
            ).detach().cpu().numpy()
        )
        vargrad_list.append(
            noise_tunnel_inst.attribute(
                inputs, target=labels.detach().cpu().numpy().tolist(), nt_type='vargrad', **saliency_params['VarGrad']
            ).detach().cpu().numpy()
        )

    label_list = np.concatenate(label_list)
    output_list = np.concatenate(output_list)

    input_list = np.concatenate(input_list)
    gradient_list = np.concatenate(gradient_list)
    smoothgrad_list = np.concatenate(smoothgrad_list)
    smoothgrad_sq_list = np.concatenate(smoothgrad_sq_list)
    vargrad_list = np.concatenate(vargrad_list)

    input_list = {
        i: input_list[np.where(label_list==i)]
        for i in range(output_list.shape[-1])
    }
    gradient_list = {
        i: gradient_list[np.where(label_list==i)]
        for i in range(output_list.shape[-1])
    }
    smoothgrad_list = {
        i: smoothgrad_list[np.where(label_list==i)]
        for i in range(output_list.shape[-1])
    }
    smoothgrad_sq_list = {        
        i: smoothgrad_sq_list[np.where(label_list==i)]
        for i in range(output_list.shape[-1])
    }
    vargrad_list = {
        i: vargrad_list[np.where(label_list==i)]
        for i in range(output_list.shape[-1])
    }
    return EvalRecord(input_list, label_list, output_list, gradient_list, smoothgrad_list, smoothgrad_sq_list, vargrad_list)

def to_holder(
    X: np.ndarray,
    y: np.ndarray,
    dev: str,
    bs: int,
    shuffle: bool = False
) -> Data.DataLoader | None:
    """Convert numpy array to torch data holder

    Args:
        X: Input data
        y: Label
        dev: Device string
        bs: Batch size
        shuffle: Whether to shuffle the data
    """

    if len(X) == 0:
        return None
    torchX = torch.tensor(X).float().to(dev)
    torchY = torch.tensor(y).long().to(dev)

    dataset = Data.TensorDataset(torchX, torchY)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle
    )
    return dataloader

class Status(Enum):
    """Utility class for training status"""
    DONE = 'Finished'
    PENDING = 'Pending'
    INIT = 'Initializing {}'
    EVAL = 'Evaluating {}'
    TRAIN = 'Training {}'

class TrainingPlanHolder:
    """class for storing training plan

    Contains repetition of training plan,
        each training plan is a :class:`TrainRecord` object

    Attributes:
        model_holder: :class:`ModelHolder` object
            Model holder
        dataset: :class:`Dataset` object
            Dataset for the training plan
        option: :class:`TrainingOption` object
            Training option
        train_record_list: List[:class:`TrainRecord`]
            List of training record generated by the training plan,
                used for storing training result
        interrupt: bool
            Whether the training is interrupted
        error: str | None
            Error message
        status: str
            Training status
    """
    def __init__(
        self,
        model_holder: ModelHolder,
        dataset: Dataset,
        option: TrainingOption, 
        saliency_params: dict,
    ):
        self.model_holder = model_holder
        self.dataset = dataset
        self.saliency_params = saliency_params
        self.option = option
        self.check_data()

        self.train_record_list = []
        self.interrupt = False
        self.error = None
        self.status = Status.PENDING.value
        for i in range(self.option.repeat_num):
            seed = set_seed(seed=None)
            model = self.model_holder.get_model(
                self.dataset.get_epoch_data().get_model_args()
            )
            self.train_record_list.append(
                TrainRecord(
                    repeat=i, dataset=self.dataset, model=model,
                    option=self.option, seed=seed
                )
            )

    def check_data(self) -> None:
        """Check whether the training plan is valid"""
        if not self.dataset:
            raise ValueError('No valid dataset is generated')
        if not self.option:
            raise ValueError('No valid training setting is generated')
        if not self.model_holder:
            raise ValueError('No valid model is selected')
        if not self.saliency_params:
            print('No saliency parameter is set, using default parameters.')
            params = {
                'nt_samples': 5,
                'nt_samples_batch_size': None,
                'stdevs': 1.0
            }
            self.saliency_params = {algo: params for algo in supported_saliency_methods}
        
        validate_type(self.model_holder, ModelHolder, 'model_holder')
        validate_type(self.dataset, Dataset, 'dataset')
        validate_type(self.option, TrainingOption, 'option')
        self.option.validate()

    # interact
    def train(self) -> None:
        """Train the model"""
        try:
            for i in range(self.option.repeat_num):
                self.status = Status.INIT.value.format(
                    self.train_record_list[i].get_name()
                )
                train_record = self.train_record_list[i]
                train_record.resume()
                self.train_one_repeat(train_record)
                train_record.pause()
            if self.is_finished():
                self.status = Status.DONE.value
            else:
                self.status = Status.PENDING.value
        except Exception as e:
            traceback.print_exc()
            self.error = str(e)
            self.status = Status.PENDING.value

    def get_loader(self) -> tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
        """Return the data loader for training, validation and testing"""
        bs = self.option.bs
        dev = self.option.get_device()
        trainHolder = to_holder(*self.dataset.get_training_data(), dev, bs, True)
        valHolder = to_holder(*self.dataset.get_val_data(), dev, bs)
        testHolder = to_holder(*self.dataset.get_test_data(), dev, bs)
        return trainHolder, valHolder, testHolder

    def get_eval_pair(
        self,
        train_record: TrainRecord,
        valLoader: Data.DataLoader,
        testLoader: Data.DataLoader
    ) -> tuple[torch.nn.Module, Data.DataLoader]:
        """Return the model and data loader for evaluation based on given option"""
        target_model = target_loader = None
        target_model = self.model_holder.get_model(
            self.dataset.get_epoch_data().get_model_args()
        ).to(self.option.get_device())
        target_loader = testLoader or valLoader
        if self.option.evaluation_option == TRAINING_EVALUATION.VAL_LOSS:
            model = getattr(train_record, f'best_val_{RecordKey.LOSS}_model')
            if model:
                target_model.load_state_dict(model)
            else:
                target_model = None
        elif self.option.evaluation_option == TRAINING_EVALUATION.TEST_ACC:
            model = getattr(train_record, f'best_test_{RecordKey.ACC}_model')
            if model:
                target_model.load_state_dict(model)
            else:
                target_model = None
        elif self.option.evaluation_option == TRAINING_EVALUATION.TEST_AUC:
            model = getattr(train_record, f'best_test_{RecordKey.AUC}_model')
            if model:
                target_model.load_state_dict(model)
            else:
                target_model = None
        elif self.option.evaluation_option == TRAINING_EVALUATION.LAST_EPOCH:
            target_model.load_state_dict(train_record.model.state_dict())
        else:
            raise NotImplementedError

        if target_model:
            target_model = target_model.eval()
        return target_model, target_loader

    def train_one_repeat(self, train_record: TrainRecord) -> None:
        """Train one repetition of the training plan

        Args:
            train_record: Training record for storing training result
        """
        if train_record.is_finished():
            return
        # init
        model = train_record.get_training_model(device=self.option.get_device())
        trainLoader, valLoader, testLoader = self.get_loader()
        if self.option.epoch > 0 and not trainLoader:
            raise ValueError('No Training Data')
        optimizer = train_record.optim
        criterion = train_record.criterion
        self.status = Status.TRAIN.value.format(train_record.get_name())
        # train one epoch
        while train_record.epoch < self.option.epoch:
            if self.interrupt:
                break
            self.train_one_epoch(
                model, trainLoader, valLoader, testLoader,
                optimizer, criterion, train_record
            )

        if train_record.epoch == self.option.epoch:
            self.status = Status.EVAL.value.format(train_record.get_name())
            target, target_loader = self.get_eval_pair(
                train_record, valLoader, testLoader
            )
            if target and target_loader:
                eval_record = _eval_model(target, target_loader, self.saliency_params)
                train_record.set_eval_record(eval_record)

        train_record.export_checkpoint()

    def train_one_epoch(self,
                        model: torch.nn.Module,
                        trainLoader: Data.DataLoader,
                        valLoader: Data.DataLoader,
                        testLoader: Data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        criterion: torch.nn.Module,
                        train_record: TrainRecord) -> None:
        """Train one epoch of the training plan"""
        start_time = time.time()
        running_loss = 0.0
        model.train()
        correct = 0
        total_count = 0
        train_auc = 0
        y_true, y_pred = None, None
        # train one mini batch
        for inputs, labels in trainLoader:
            if self.interrupt:
                return
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(axis=1) == labels).float().sum().item()
            if y_true is None or y_pred is None:
                y_true = labels
                y_pred = outputs
            else:
                y_true = torch.cat((y_true, labels))
                y_pred = torch.cat((y_pred, outputs))
            total_count += len(labels)
            running_loss += loss.item()

        try:
            if y_pred.size()[-1] <=2:
                train_auc = roc_auc_score(y_true.clone().detach().cpu().numpy(),
                torch.nn.functional.softmax(
                    y_pred, dim=1
                ).clone().detach().cpu().numpy()[:, 1])
            else:
                train_auc = roc_auc_score(y_true.clone().detach().cpu().numpy(),
                torch.nn.functional.softmax(
                    y_pred, dim=1
                ).clone().detach().cpu().numpy(), multi_class='ovr')
        except Exception:
            # first few epochs in binary classification
            # might not be able to compute score
            pass


        running_loss /= len(trainLoader)
        train_acc = correct / total_count * 100
        train_record.update_train({
            RecordKey.LOSS: running_loss,
            RecordKey.ACC: train_acc,
            RecordKey.AUC: train_auc
        })

        if valLoader:
            test_result = _test_model(model, valLoader, criterion)
            train_record.update_eval(test_result)

        trainingTime = time.time() - start_time

        if testLoader:
            test_result = _test_model(model, testLoader, criterion)
            train_record.update_test(test_result)

        train_record.update_statistic({
            TrainRecordKey.LR: optimizer.param_groups[0]['lr'],
            TrainRecordKey.TIME: trainingTime
        })
        train_record.step()
        if (
            self.option.checkpoint_epoch and
            train_record.get_epoch() % self.option.checkpoint_epoch == 0
        ):
            train_record.export_checkpoint()

    def set_interrupt(self) -> None:
        """Set the training plan to be interrupted"""
        self.interrupt = True

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag and error status"""
        self.error = None
        self.interrupt = False

    # getter
    def get_name(self) -> str:
        """Return the name of the training plan"""
        return self.dataset.get_name()

    def get_dataset(self) -> Dataset:
        """Get the dataset of the training plan"""
        return self.dataset

    def get_plans(self) -> list[TrainRecord]:
        """Get the training records of the training plan"""
        return self.train_record_list

    def get_saliency_params(self) -> dict:
        """Return the saliency computation parameters"""
        return self.saliency_params
    
    # setter
    def set_saliency_params(self, saliency_params)-> None:
        """Set the saliency computation parameters"""
        self.saliency_params = saliency_params
        for i in range(self.option.repeat_num):
            train_record = self.train_record_list[i]
            trainLoader, valLoader, testLoader = self.get_loader()
            target, target_loader = self.get_eval_pair(
                train_record, valLoader, testLoader
            )
            if target is not None: # model is trained
                eval_record = _eval_model(target, target_loader, self.saliency_params)
                self.train_record_list[i].set_eval_record(eval_record)

    # status
    def get_training_status(self) -> str:
        """Return the training status"""
        if self.error:
            return self.error
        return self.status

    def get_training_repeat(self) -> int:
        """Return the index of the current training repetition"""
        for i in range(self.option.repeat_num):
            if not self.train_record_list[i].is_finished():
                break
        return i

    def get_training_epoch(self) -> int:
        """Return the current epoch of the training plan"""
        return self.train_record_list[self.get_training_repeat()].get_epoch()

    def get_training_evaluation(self) -> tuple:
        """Return the evaluation result of the training plan

        Return:
            Tuple of lr, train_loss, train_acc, train_auc, val_loss, val_acc
        """
        record = self.train_record_list[self.get_training_repeat()]

        lr = train_loss = train_acc = train_auc = \
            val_loss = val_acc = val_auc = '-'
        if len(record.train[TrainRecordKey.LR]) > 0:
            lr = record.train[TrainRecordKey.LR][-1]
        if len(record.train[TrainRecordKey.LOSS]) > 0:
            train_loss = record.train[TrainRecordKey.LOSS][-1]
        if len(record.train[TrainRecordKey.AUC]) > 0:
            train_auc = record.train[TrainRecordKey.AUC][-1]
        if len(record.train[TrainRecordKey.ACC]) > 0:
            train_acc = record.train[TrainRecordKey.ACC][-1]
        if len(record.val[RecordKey.LOSS]) > 0:
            val_loss = record.val[RecordKey.LOSS][-1]
        if len(record.val[RecordKey.ACC]) > 0:
            val_acc = record.val[RecordKey.ACC][-1]
        if len(record.val[RecordKey.AUC]) > 0:
            val_auc = record.val[RecordKey.AUC][-1]
        return lr, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc

    def is_finished(self) -> bool:
        """Return whether the training plan is finished"""
        return self.train_record_list[-1].is_finished()

    def get_epoch_progress_text(self) -> str:
        """Return the progress text of the training plan"""
        total = 0
        for train_record in self.train_record_list:
            total += train_record.get_epoch()
        return f"{total} / {self.option.epoch * self.option.repeat_num}"
