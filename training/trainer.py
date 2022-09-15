from enum import Enum
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
from copy import deepcopy

class TRAINING_EVALUATION(Enum):
    VAL_LOSS = 'Best validation loss'
    TEST_ACC = 'Best testing performance'
    LAST_EPOCH = 'Last Epoch'

class TrainRecord:
    def __init__(self, model, optim, repeat):
        self.model = model
        self.optim = optim
        #
        self.best_val_loss_model = None
        self.best_val_acc_model = None
        self.best_test_acc_model = None
        #
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
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
    
    def get_loss_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
        from matplotlib import pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()

        training_loss_list = self.train['loss']
        val_loss_list = self.val['loss']
        test_loss_list = self.test['loss']
        
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
        from matplotlib import pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        training_acc_list = self.train['acc']
        val_acc_list = self.val['acc']
        test_acc_list = self.test['acc']

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
        from matplotlib import pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.clf()
        
        lr_list = self.lr

        plt.plot(lr_list, 'g')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('lr')
        return fig

def testModel(model, dataLoader, criterion):
    model.eval()
    running_loss = 0.0
    total_count = 0
    correct = 0
    for inputs, labels in dataLoader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        correct += (outputs.argmax(axis=1) == labels).float().sum().item()
        total_count += len(labels)
        running_loss += loss.item()

    running_loss /= len(dataLoader)
    acc = correct / total_count * 100
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

class Trainer:
    def __init__(self, model_holder, dataset, option):
        self.model_holder = model_holder
        self.dataset = dataset
        self.option = option
        self.train_record_list = []
        self.interrupt = False

    def train(self):
        for i in range(self.option.repeat_num):
            if i == len(self.train_record_list):
                model = self.model_holder.get_model(self.dataset.get_args())
                optim = self.option.get_optim(model)
                self.train_record_list.append(TrainRecord(model, optim, repeat=i))
            self.train_repeat( self.train_record_list[i] )

    def get_loader(self):
        bs = self.option.bs
        dev = self.option.get_device()
        trainHolder = to_holder(*self.dataset.get_training_data(), dev, bs, True)
        valHolder = to_holder(*self.dataset.get_val_data(), dev, bs)
        testHolder = to_holder(*self.dataset.get_test_data(), dev, bs)
        return trainHolder, valHolder, testHolder

    def train_repeat(self, train_record):
        model = train_record.model.to(self.option.get_device())
        trainLoader, valLoader, testLoader = self.get_loader()
        optimizer = train_record.optim
        criterion = train_record.criterion
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
            train_record.finished = True

    def set_interrupt(self):
        self.interrupt = True

    def clear_interrupt(self):
        self.interrupt = False

    def get_training_repeat(self):
        return len(self.train_record_list)

    def get_training_epoch(self):
        if len(self.train_record_list) == 0:
            return 0
        return self.train_record_list[-1].epoch

    def get_training_evaluation(self):
        if len(self.train_record_list) == 0:
            return []
        record = self.train_record_list[-1]
        return record.lr[-1] if len(record.lr) > 0 else '-', record.train['loss'][-1] if len(record.train['loss']) > 0 else '-', record.train['acc'][-1] if len(record.train['acc']) > 0 else '-', record.val['loss'][-1] if len(record.val['loss']) > 0 else '-', record.val['acc'][-1] if len(record.val['acc']) > 0 else '-'
    
    def is_finished(self):
        return len(self.train_record_list) == self.option.repeat_num and self.train_record_list[-1].epoch == self.option.epoch

    def get_plans(self):
        return self.train_record_list
