import os
import shutil
import torch
import time
from matplotlib import pyplot as plt
from copy import deepcopy
from ...utils import set_random_state, get_random_state
from .eval import calculate_confusion

class TrainRecord:
    def __init__(self, repeat, dataset, model, option, seed):
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

    def resume(self):
        set_random_state(self.random_state)

    def pause(self):
        self.random_state = get_random_state()

    def create_dir(self):
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

    def get_name(self):
        return f"Repeat-{self.repeat}"

    def get_epoch(self):
        return self.epoch

    def get_training_model(self, device):
        return self.model.to(device)

    def is_finished(self):
        return self.get_epoch() >= self.option.epoch and self.eval_record is not None
    #
    def append_record(self, val, arr):
        while len(arr) < self.epoch:
            arr.append(None)
        if len(arr) > self.epoch:
            arr[self.epoch] = val
        elif len(arr) == self.epoch:
            arr.append(val)

    def update_eval(self, val_acc, val_auc, val_loss):
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
    
    def update_test(self, test_acc, test_auc, test_loss):
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
    def update_train(self, train_acc, train_auc, running_loss):
        self.append_record(train_acc, self.train['acc'])
        self.append_record(train_auc, self.train['auc'])
        self.append_record(running_loss, self.train['loss'])
    
    def step(self, trainingTime, lr):
        self.append_record(trainingTime, self.train['time'])
        self.append_record(lr, self.train['lr'])
        self.epoch += 1

    def set_eval_record(self, eval_record):
        self.eval_record = eval_record
    # 
    def export_checkpoint(self):
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

    def get_auc_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100): ## TODO
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
    
    def get_lr_figure(self, fig=None, figsize=(6.4, 4.8), dpi=100):
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
    def get_acc(self):
        if not self.eval_record:
            return None
        return self.eval_record.get_acc()
    
    def get_auc(self):
        if not self.eval_record:
            return None
        return self.eval_record.get_auc()

    def get_kappa(self):
        if not self.eval_record:
            return None
        return self.eval_record.get_kappa()

    def get_eval_record(self):
        return self.eval_record
