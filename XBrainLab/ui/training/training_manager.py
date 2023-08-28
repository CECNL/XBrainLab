import tkinter as tk
import tkinter.messagebox
from ..widget import EditableTreeView, PlotFigureWindow
from ..base import TopWindow, InitWindowValidateException, ValidateException
from ..script import Script
from XBrainLab.training import Trainer
from XBrainLab.visualization import PlotType

##
class TrainingManagerWindow(TopWindow):
    def __init__(self, parent, trainer):
        super().__init__(parent, 'Training Manager')
        self.trainer = trainer
        self.check_data()
        self.training_plan_holders = trainer.get_training_plan_holders()
        self.script_history = Script()

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
        
        if trainer.is_running():
            self.start_training()

    def check_data(self):
        if type(self.trainer) != Trainer:
            raise InitWindowValidateException(self, 'No valid training plan is generated')
        if type(self.trainer.get_training_plan_holders()) != list:
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
        module = PlotFigureWindow(self, self.training_plan_holders, PlotType.LOSS)
        self.script_history += module.get_script_history()

    def plot_acc(self):
        module = PlotFigureWindow(self, self.training_plan_holders, PlotType.ACCURACY)
        self.script_history += module.get_script_history()

    def plot_lr(self):
        module = PlotFigureWindow(self, self.training_plan_holders, PlotType.LR)
        self.script_history += module.get_script_history()

    # train
    def start_training(self):
        self.start_btn.config(state=tk.DISABLED)
        if not self.trainer.is_running():
            self.trainer.run(interact=True)
            self.script_history.add_cmd("study.train(interact=True)")
            
        self.training_loop()

    def stop_training(self):
        if not self.trainer.is_running():
            raise ValidateException(window=self, message='No training is in progress')
        try:
            not self.trainer.set_interrupt()
        except:
            pass

    def finish_training(self):
        self.start_btn.config(state=tk.NORMAL)
        self.status_bar.config(text='IDLE')
        self.update_table()
        if not self.window_exist:
            return
        tk.messagebox.showinfo('Success', 'Training has stopped', parent=self)

    def training_loop(self):
        if not self.window_exist:
            return
        self.update_table()
        if not self.trainer.is_running():
            self.finish_training()
            return
        self.after(100, self.training_loop)

    def update_table(self):
        plan_tree = self.plan_tree
        def get_table_values(plan):
            return (plan.get_name(), plan.get_training_status(), plan.get_training_epoch(), *plan.get_training_evaluation())
        if len(plan_tree.get_children()) == 0:
            # for initialization
            for training_trainer in self.training_plan_holders:
                update_node = plan_tree.insert("", index='end', values=get_table_values(training_trainer))
        else:
            # for updating
            for idx, training_plan in enumerate(self.training_plan_holders):
                item = plan_tree.get_children()[idx]
                plan_tree.item(item, values=get_table_values(training_plan))

        self.status_bar.config(text=self.trainer.get_progress_text())

    def _get_script_history(self):
        return self.script_history