import torch
import torch.utils.data as Data
import time
from copy import deepcopy
import numpy as np
from enum import Enum
import traceback
from captum.attr import Saliency

from ..utils import validate_type, set_seed, set_random_state, get_random_state
from ..dataset import Dataset
from .option import TrainingOption, TRAINING_EVALUATION
from .record import TrainRecord, EvalRecord
from . import ModelHolder


def test_model(model, dataLoader, criterion):
    model.eval()
    
    running_loss = 0.0
    total_count = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataLoader:
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            correct += (outputs.argmax(axis=1) == labels).float().sum().item()
            total_count += len(labels)

    running_loss /= len(dataLoader)
    acc = correct / total_count * 100
    return acc, running_loss

def eval_model(model, dataLoader, criterion):
    model.eval()
    
    output_list = []
    label_list = []
    
    ## gradient_list = None
    gradient_list = []
    saliency_inst = Saliency(model)

    for inputs, labels in dataLoader:
        # inputs.requires_grad=True
        inputs.requires_grad = False
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())

        ## if gradient_list is None:
        ##     gradient_list = {i: [] for i in range(outputs.shape[-1])}
        ## for i in gradient_list:
        ##     inputs.grad = None
        ##     for output in outputs:
        ##         output[i].backward(retain_graph=True)
        ##     gradient_list[i].append(inputs.grad.detach().cpu().numpy())
        inputs.requires_grad=True
        gradient_list.append(saliency_inst.attribute(inputs,target=labels.detach().cpu().numpy().tolist()).detach().cpu().numpy().squeeze())

    label_list = np.concatenate(label_list)
    output_list = np.concatenate(output_list)
   
    ## for i in gradient_list:
    ##     gradient_list[i] = np.concatenate(gradient_list[i])
    gradient_list = np.concatenate(gradient_list)
    ## gradient_list = {i:gradient_list[np.where(output_list.argmax(axis=-1)==i)] for i in range(output_list.shape[-1])}
    gradient_list = {i:gradient_list[np.where(label_list==i)] for i in range(output_list.shape[-1])}
    return EvalRecord(label_list, output_list, gradient_list)

def to_holder(X, y, dev, bs, shuffle=False):
    if len(X) == 0:
        return None
    torchX = torch.tensor(X).float().to(dev)
    torchY = torch.tensor(y).long().to(dev)

    dataset = Data.TensorDataset(torchX, torchY)
    dataloader = Data.DataLoader(dataset,
                              batch_size=bs,
                              shuffle=shuffle
                              )
    return dataloader

class Status(Enum):
    DONE = 'Finished'
    PENDING = 'Pending'
    INIT = 'Initializing {}'
    EVAL = 'Evaluating {}'
    TRAIN = 'Training {}'

class TrainingPlanHolder:
    def __init__(self, model_holder, dataset, option):
        self.model_holder = model_holder
        self.dataset = dataset
        self.option = option
        self.check_data()

        self.train_record_list = []
        self.interrupt = False
        self.error = None
        self.status = Status.PENDING.value
        for i in range(self.option.repeat_num):
            seed = set_seed(seed=None)
            model = self.model_holder.get_model(self.dataset.get_epoch_data().get_model_args())
            self.train_record_list.append(TrainRecord(repeat=i, dataset=self.dataset, model=model, option=self.option, seed=seed))
        if len(self.train_record_list) == 0:
            raise ValueError('Invalid training settings.')
          
    def check_data(self):
        if not self.dataset:
            raise ValueError('No valid dataset is generated')
        if not self.option:
            raise ValueError('No valid training setting is generated')
        if not self.model_holder:
            raise ValueError('No valid model is selected')
        validate_type(self.model_holder, ModelHolder, 'model_holder')
        validate_type(self.dataset, Dataset, 'dataset')
        validate_type(self.option, TrainingOption, 'option')
        self.option.validate()

    # interact
    def train(self):
        try:
            for i in range(self.option.repeat_num):
                self.status = Status.INIT.value.format(self.train_record_list[i].get_name())
                train_record = self.train_record_list[i]
                train_record.resume()
                self.train_one_repeat( train_record )
                train_record.pause()
            if self.is_finished():
                self.status = Status.DONE.value
            else:
                self.status = Status.PENDING.value
        except Exception as e:
            traceback.print_exc()
            self.error = str(e)
            self.status = Status.PENDING.value
    
    def get_loader(self):
        bs = self.option.bs
        dev = self.option.get_device()
        trainHolder = to_holder(*self.dataset.get_training_data(), dev, bs, True)
        valHolder = to_holder(*self.dataset.get_val_data(), dev, bs)
        testHolder = to_holder(*self.dataset.get_test_data(), dev, bs)
        return trainHolder, valHolder, testHolder
    
    def get_eval_pair(self, train_record, trainLoader, valLoader, testLoader):
        target = target_loader = None
        target = self.model_holder.get_model(self.dataset.get_epoch_data().get_model_args()).to(self.option.get_device())
        target_loader = testLoader or valLoader
        if self.option.evaluation_option == TRAINING_EVALUATION.VAL_LOSS:
            if train_record.best_val_loss_model:
                target.load_state_dict(train_record.best_val_loss_model)
            else:
                target = None
        elif self.option.evaluation_option == TRAINING_EVALUATION.TEST_ACC:
            if train_record.best_test_acc_model:
                target.load_state_dict(train_record.best_test_acc_model)
            else:
                target = None
        elif self.option.evaluation_option == TRAINING_EVALUATION.LAST_EPOCH:
            target.load_state_dict(train_record.model.state_dict())
        if target:
            target = target.eval()
        return target, target_loader

    def train_one_repeat(self, train_record):
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
            cur_epoch = train_record.epoch + 1

            start_time = time.time()
            running_loss = 0.0
            model.train()
            correct = 0
            total_count = 0
            # train one mini batch
            for inputs, labels in trainLoader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (outputs.argmax(axis=1) == labels).float().sum().item()
                total_count += len(labels)
                running_loss += loss.item()
            
            running_loss /= len(trainLoader)
            trian_acc = correct / total_count * 100
            train_record.update_train(trian_acc, running_loss)
            
            if valLoader:
                val_acc, val_loss = test_model(model, valLoader, criterion)
                train_record.update_eval(val_acc, val_loss)
            
            trainingTime = time.time() - start_time

            if testLoader:
                test_acc, test_loss = test_model(model, testLoader, criterion)
                train_record.update_test(test_acc, test_loss)
            
            train_record.step(trainingTime, self.option.lr)
            if self.option.checkpoint_epoch and train_record.epoch % self.option.checkpoint_epoch == 0:
                train_record.export_checkpoint()
        
        if train_record.epoch == self.option.epoch:
            self.status = Status.EVAL.value.format(train_record.get_name())
            target, target_loader = self.get_eval_pair(train_record, trainLoader, valLoader, testLoader)
            if target and target_loader:
                eval_record = eval_model(target, target_loader, criterion)
                train_record.set_eval_record(eval_record)

        train_record.export_checkpoint()
     
    def set_interrupt(self):
        self.interrupt = True

    def clear_interrupt(self):
        self.error = None
        self.interrupt = False

    # getter
    def get_name(self):
        return self.dataset.get_name()
    
    def get_dataset(self):
        return self.dataset

    def get_plans(self):
        return self.train_record_list
    
    # status
    def get_training_status(self):
        if self.error:
            return self.error
        return self.status

    def get_training_repeat(self):
        for i in range(self.option.repeat_num):
            if not self.train_record_list[i].is_finished():
                break
        return i

    def get_training_epoch(self):
        return self.train_record_list[self.get_training_repeat()].epoch

    def get_training_evaluation(self):
        record = self.train_record_list[self.get_training_repeat()]
        lr = record.train['lr'][-1] if len(record.train['lr']) > 0 else '-'
        train_loss = record.train['loss'][-1] if len(record.train['loss']) > 0 else '-'
        train_acc = record.train['acc'][-1] if len(record.train['acc']) > 0 else '-'
        val_loss = record.val['loss'][-1] if len(record.val['loss']) > 0 else '-'
        val_acc = record.val['acc'][-1] if len(record.val['acc']) > 0 else '-'
        return lr, train_loss, train_acc, val_loss, val_acc
    
    def is_finished(self):
        return self.train_record_list[-1].is_finished()
    
    def get_epoch_progress_text(self):
        total = 0
        for train_record in self.train_record_list:
            total += train_record.epoch
        return f"{total} / {self.option.epoch * self.option.repeat_num}"
        

        