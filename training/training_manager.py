import tkinter as tk
import tkinter.messagebox
from ..widget import EditableTreeView, PlotFigureWindow, PlotType
from ..base import TopWindow, InitWindowValidateException, ValidateException

import threading, time

class TrainingManagerJob():
    def __init__(self, trainers):
        self.finished = False
        self.interrupt = False
        self.progress_text = 'initializing'
        self.trainers = trainers
    
    def set_interrupt(self):
        self.interrupt = True
        for trainer in self.trainers:
            trainer.set_interrupt()

    def job(self):
        for trainer in self.trainers:
            self.progress_text = f'Now training: {trainer.get_name()}'
            if self.interrupt:
                break
            trainer.train()
        self.finished = True
        TrainingManagerWindow.task = None

    def run(self):
        threading.Thread(target=self.job).start()

    def is_finished(self):
        return self.finished
##
class TrainingManagerWindow(TopWindow):
    task = None
    def __init__(self, parent, trainers):
        super().__init__(parent, 'Training Manager')
        self.trainers = trainers
        self.check_data()
        columns = ('Plan name', 'Status', 'Epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc')
        plan_tree = EditableTreeView(self, columns=columns, show='headings')

        status_bar = tk.Label(self, text='IDLE')

        button_frame = tk.Frame(self)
        start_btn = tk.Button(button_frame, text='start', command=self.start_training)
        start_btn.pack(side=tk.LEFT)
        tk.Button(button_frame, text='stop', command=self.stop_training).pack(side=tk.LEFT)

        plan_tree.pack(fill=tk.BOTH, expand=True)
        status_bar.pack(fill=tk.X)
        button_frame.pack()

        self.start_btn = start_btn
        self.status_bar = status_bar
        self.plan_tree = plan_tree

        self.config_menu()
        self.update_table()
        
        if TrainingManagerWindow.task:
            self.start_training()

    def check_data(self):
        if type(self.trainers) != list:
            raise InitWindowValidateException(self, 'No valid training plan is generated')

    # plot
    def config_menu(self):
        menu = tk.Menu(self, tearoff=0)
        self.config(menu=menu)

        plot_menu = tk.Menu(menu, tearoff=0)
        plot_menu.add_command(label="Loss", command=self.plot_loss)
        plot_menu.add_command(label="Accuracy", command=self.plot_acc)
        plot_menu.add_command(label="Learning Rate", command=self.plot_lr)

        menu.add_cascade(label='Plot', menu=plot_menu)

    def plot_loss(self):
        PlotFigureWindow(self, self.trainers, PlotType.LOSS)

    def plot_acc(self):
        PlotFigureWindow(self, self.trainers, PlotType.ACCURACY)

    def plot_lr(self):
        PlotFigureWindow(self, self.trainers, PlotType.LR)

    # train
    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        if TrainingManagerWindow.task is None:     # no running task
            for training_trainer in self.trainers:
                training_trainer.clear_interrupt()

            TrainingManagerWindow.task = TrainingManagerJob(self.trainers)
            TrainingManagerWindow.task.run()
        self.training_loop()

    def stop_training(self):
        if TrainingManagerWindow.task is None:
            raise ValidateException(window=self, message='No training is in progress')
        try:
            TrainingManagerWindow.task.set_interrupt()
        except:
            pass

    def finish_training(self):
        TrainingManagerWindow.task = None
        self.start_btn.config(state=tk.NORMAL)
        self.status_bar.config(text='IDLE')
        self.update_table()
        if not self.winfo_exists():
            return
        tk.messagebox.showinfo('Success', 'Training has stopped', parent=self)

    def training_loop(self):
        if self.winfo_exists() == 0:
            return
        if not TrainingManagerWindow.task:
            self.finish_training()
            return
        if TrainingManagerWindow.task.is_finished():
            self.finish_training()
            return
        self.update_table()
        self.after(100, self.training_loop)

    def update_table(self):
        plan_tree = self.plan_tree
        def get_table_values(plan):
            return (plan.get_name(), plan.get_training_status(), plan.get_training_epoch(), *plan.get_training_evaluation())
        if len(plan_tree.get_children()) == 0:
            # for initialization
            for training_trainer in self.trainers:
                update_node = plan_tree.insert("", index='end', values=get_table_values(training_trainer))
        else:
            # for updating
            for idx, training_plan in enumerate(self.trainers):
                item = plan_tree.get_children()[idx]
                plan_tree.item(item, values=get_table_values(training_plan))

        if TrainingManagerWindow.task:
            self.status_bar.config(text=TrainingManagerWindow.task.progress_text)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    trainers = []
    
    option = torch.load('data/data_structure/setting')
    model_holder = torch.load('data/data_structure/model')
    datasets = torch.load('data/data_structure/splitted')
    for dataset in datasets:
        trainers.append(Trainer(option, model_holder, dataset))

    window = TrainingManagerWindow(root, trainers)

    print (window.get_result())
    root.destroy()