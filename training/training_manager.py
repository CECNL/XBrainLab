import tkinter as tk
import tkinter.messagebox
from ..widget import EditableTreeView
from ..base import TopWindow
from .training_plot import TrainingPlotWindow, TrainingPlotType

import threading
class TrainingManagerJob():
    def __init__(self, training_plan_holders):
        self.finished = False
        self.interrupt = False
        self.progress_text = 'initializing'
        self.training_plan_holders = training_plan_holders
    
    def set_interrupt(self):
        self.interrupt = True
        for plan_holder in self.training_plan_holders:
            plan_holder.set_interrupt()

    def job(self):
        for plan_holder in self.training_plan_holders:
            self.progress_text = f'Now training: {plan_holder.get_name()}'
            if self.interrupt:
                break
            plan_holder.train(job=self)
        self.finished = True

    def run(self):
        threading.Thread(target=self.job).start()

    def is_finished(self):
        return self.finished
##
class TrainingManagerWindow(TopWindow):
    task = None
    def __init__(self, parent, training_plan_holders):
        super().__init__(parent, 'Training Manager')
        self.training_plan_holders = training_plan_holders
        if not self.check_data():
            return
        columns = ('Plan name', 'Status', 'Epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc')
        plan_tree = EditableTreeView(self, columns=columns)

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
        if type(self.training_plan_holders) != list:
            self.withdraw()
            tk.messagebox.showerror(parent=self, title='Error', message='No valid training plan is generated')
            self.destroy()
            return False
        return True

    def config_menu(self):
        menu = tk.Menu(self, tearoff=0)
        self.config(menu=menu)

        plot_menu = tk.Menu(menu, tearoff=0)
        plot_menu.add_command(label="Loss", command=self.plot_loss)
        plot_menu.add_command(label="Accuracy", command=self.plot_acc)
        plot_menu.add_command(label="Learning Rate", command=self.plot_lr)

        menu.add_cascade(label='Plot', menu=plot_menu)

    def plot_loss(self):
        TrainingPlotWindow(self, self.training_plan_holders, TrainingPlotType.LOSS)

    def plot_acc(self):
        TrainingPlotWindow(self, self.training_plan_holders, TrainingPlotType.ACCURACY)

    def plot_lr(self):
        TrainingPlotWindow(self, self.training_plan_holders, TrainingPlotType.LR)

    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        if TrainingManagerWindow.task is None:     # no running task
            for training_plan_holder in self.training_plan_holders:
                training_plan_holder.clear_interrupt()

            TrainingManagerWindow.task = TrainingManagerJob(self.training_plan_holders)
            TrainingManagerWindow.task.run()
        self.training_loop()

    def stop_training(self):
        if TrainingManagerWindow.task is None:
            tk.messagebox.showerror('Error', 'No training is in progress', parent=self)
            return
        try:
            TrainingManagerWindow.task.set_interrupt()
        except:
            pass

    def finish_training(self):
        TrainingManagerWindow.task = None
        self.start_btn.config(state=tk.NORMAL)
        self.status_bar.config(text='IDLE')
        self.update_table()
        tk.messagebox.showinfo('Success', 'Training has stopped', parent=self)

    def training_loop(self):
        if self.winfo_exists() == 0:
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
            for training_plan_holder in self.training_plan_holders:
                update_node = plan_tree.insert("", index='end', values=get_table_values(training_plan_holder))
        else:
            # for updating
            for idx, training_plan in enumerate(self.training_plan_holders):
                item = plan_tree.get_children()[idx]
                plan_tree.item(item, values=get_table_values(training_plan))

        if TrainingManagerWindow.task:
            self.status_bar.config(text=TrainingManagerWindow.task.progress_text)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    training_plan_holders = []
    
    option = torch.load('data/data_structure/setting')
    model_holder = torch.load('data/data_structure/model')
    datasets = torch.load('data/data_structure/splitted')
    for dataset in datasets:
        training_plan_holders.append(TrainingPlan(option, model_holder, dataset))

    window = TrainingManagerWindow(root, training_plan_holders)

    print (window.get_result())
    root.destroy()