import torch
import torch.nn as nn
import torch.utils.data as Data
import time
from matplotlib import pyplot as plt
from copy import deepcopy
from ..base import InitValidateException
from .training_setting import TRAINING_EVALUATION
import numpy as np
from enum import Enum
import traceback

def calculate_confusion(output, label):
    classNum = len(np.unique(label))
    confusion = np.zeros((classNum,classNum), dtype=np.uint32)
    output = output.argmax(axis=1)
    for ground_truth in range(classNum):
        for predict in range(classNum):
            confusion[ground_truth][predict] = (output[label == ground_truth] == predict).sum()
    return confusion

class EvalRecord:
    def __init__(self, label, output, gradient):
        self.label = label
        self.output = output
        self.gradient = gradient
    
    def get_acc(self):
        return sum(self.output.argmax(axis=1) == self.label) / len(self.label)

    def get_kappa(self):
        confusion = calculate_confusion(self.output, self.label)
        classNum = len(confusion)
        P0 = np.diagonal(confusion).sum() / confusion.sum()
        Pe = sum([confusion[:,i].sum() * confusion[i].sum() for i in range(classNum)]) / (confusion.sum() * confusion.sum())
        return (P0 - Pe) / (1 - Pe)

class TrainRecord:
    def __init__(self, model, dataset, optim, repeat):
        self.model = model
        self.dataset = dataset
        self.optim = optim
        #
        self.eval_record = None
        self.best_val_loss_model = None
        self.best_val_acc_model = None
        self.best_test_acc_model = None
        # 
        self.train = {'loss': [], 'acc': [], 'time': []}
        self.val = {'loss': [], 'acc': []}
        self.test = {'loss': [], 'acc': [], 'best_val_loss': torch.inf, 'best_val_acc': -1, 'best_test_acc': -1,
                                'best_val_loss_epoch': None, 'best_val_acc_epoch': None, 'best_test_acc_epoch': None}
        self.lr = []
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 0
        self.repeat = repeat
        self.finished = False

    def get_name(self):
        return f"Repeat {self.repeat}"

    def is_finished(self):
        return self.finished
    
    # figure
    def get_loss_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
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

    def get_acc_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
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
    
    def get_lr_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        lr_list = self.lr
        if len(lr_list) == 0:
            return None

        plt.plot(lr_list, 'g')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('lr')
        return fig

    def get_confusion_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
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
        res = ax.imshow(confusion, cmap='Blues', interpolation='nearest')
        for x in range(classNum):
            for y in range(classNum):
                ax.annotate(str(confusion[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
        cb = fig.colorbar(res)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        labels = [self.dataset.data_holder.label_map[l] for l in range(classNum)]
        plt.xticks(range(classNum), labels)
        plt.yticks(range(classNum), labels)

        return fig
    
    # get evaluate
    def get_acc(self):
        if not self.eval_record:
            return None
        return self.eval_record.get_acc()

    def get_kappa(self):
        if not self.eval_record:
            return None
        return self.eval_record.get_kappa()

    def get_eval_record(self):
        return self.eval_record

def testModel(model, dataLoader, criterion, return_output=False):
    model.eval()
    output_list = []
    label_list = []
    gradient_list = None
    running_loss = 0.0
    total_count = 0
    correct = 0
    for inputs, labels in dataLoader:
        inputs.requires_grad=True
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        correct += (outputs.argmax(axis=1) == labels).float().sum().item()
        total_count += len(labels)
        running_loss += loss.item()

        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())
        if return_output:
            if gradient_list is None:
                gradient_list = {i: [] for i in range(outputs.shape[-1])}
            for i in gradient_list:
                inputs.grad = None
                for output in outputs:
                    output[i].backward(retain_graph=True)
                gradient_list[i].append(inputs.grad.detach().cpu().numpy())

    running_loss /= len(dataLoader)
    acc = correct / total_count * 100
    if return_output:
        for i in gradient_list:
            gradient_list[i] = np.concatenate(gradient_list[i])
        return EvalRecord(np.concatenate(label_list), np.concatenate(output_list), gradient_list)
    return acc, running_loss

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

class Trainer:
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
            model = self.model_holder.get_model(self.dataset.get_data_holder().get_args())
            optim = self.option.get_optim(model)
            self.train_record_list.append(TrainRecord(model, self.dataset, optim, repeat=i))
        if len(self.train_record_list) == 0:
            raise InitValidateException('Invalid training settings.')
    
    def check_data(self):
        if not self.dataset:
            raise InitValidateException('No valid dataset is generated')
        if not self.option:
            raise InitValidateException('No valid training setting is generated')
        if not self.model_holder:
            raise InitValidateException('No valid model is selected')

    # interact
    def train(self):
        try:
            for i in range(self.option.repeat_num):
                self.status = Status.INIT.value.format(self.train_record_list[i].get_name())
                self.train_one_repeat( self.train_record_list[i] )
            self.status = Status.DONE.value
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
    
    def get_eval_pair(self, train_record):
        target = target_loader = None
        trainLoader, valLoader, testLoader = self.get_loader()
        target = self.model_holder.get_model(self.dataset.get_data_holder().get_args()).to(self.option.get_device())
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
            target = train_record.model
        return target, target_loader

    def train_one_repeat(self, train_record):
        if train_record.is_finished():
            return
        model = train_record.model.to(self.option.get_device())
        trainLoader, valLoader, testLoader = self.get_loader()
        if not trainLoader:
            raise ValueError('No Training Data')
        optimizer = train_record.optim
        criterion = train_record.criterion
        self.status = Status.TRAIN.value.format(train_record.get_name())
        while train_record.epoch < self.option.epoch:
            if self.interrupt:
                break
            cur_epoch = train_record.epoch + 1

            start_time = time.time()
            running_loss = 0.0
            model.train()
            total_count = 0
            correct = 0
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
            trainingTime = time.time() - start_time
            tacc = correct / total_count * 100
            
            train_record.train['loss'].append(running_loss)
            train_record.train['acc'].append(tacc)
            train_record.train['time'].append(trainingTime)
            
            if valLoader:
                val_acc, val_loss = testModel(model, valLoader, criterion)
                train_record.val['acc'].append(val_acc)
                train_record.val['loss'].append(val_loss)

                if val_loss <= train_record.test['best_val_loss']:
                    earlyEpoch = cur_epoch
                    train_record.test['best_val_loss'] = val_loss
                    train_record.test['best_val_loss_epoch'] = cur_epoch
                    train_record.best_val_loss_model = deepcopy(model.state_dict())

                if val_acc >= train_record.test['best_val_acc']:
                    train_record.test['best_val_acc'] = val_acc
                    train_record.test['best_val_acc_epoch'] = cur_epoch
                    train_record.best_val_acc_model = deepcopy(model.state_dict())

            if testLoader:
                test_acc, test_loss = testModel(model, testLoader, criterion)
                train_record.test['acc'].append(test_acc)
                train_record.test['loss'].append(test_loss)
            
                if test_acc >= train_record.test['best_test_acc']:
                    train_record.test['best_test_acc'] = test_acc
                    train_record.test['best_test_acc_epoch'] = cur_epoch
                    train_record.best_test_acc_model = deepcopy(model.state_dict())
            train_record.lr.append(self.option.lr)
            train_record.epoch += 1
        
        if train_record.epoch == self.option.epoch:
            self.status = Status.EVAL.value.format(train_record.get_name())
            target, target_loader = self.get_eval_pair(train_record)
            if target and target_loader:
                train_record.eval_record = testModel(target, target_loader, criterion, return_output=True)

            train_record.finished = True
     
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
        return self.status

    def get_training_repeat(self):
        for i in range(self.option.repeat_num):
            if not self.train_record_list[i].is_finished():
                return i
        return self.option.repeat_num - 1

    def get_training_epoch(self):
        return self.train_record_list[self.get_training_repeat()].epoch

    def get_training_evaluation(self):
        record = self.train_record_list[self.get_training_repeat()]
        lr = record.lr[-1] if len(record.lr) > 0 else '-'
        train_loss = record.train['loss'][-1] if len(record.train['loss']) > 0 else '-'
        train_acc = record.train['acc'][-1] if len(record.train['acc']) > 0 else '-'
        val_loss = record.val['loss'][-1] if len(record.val['loss']) > 0 else '-'
        val_acc = record.val['acc'][-1] if len(record.val['acc']) > 0 else '-'
        return lr, train_loss, train_acc, val_loss, val_acc
    
    def is_finished(self):
        return self.train_record_list[-1].is_finished()