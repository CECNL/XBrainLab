import tkinter as tk

from torchinfo import summary

from ..base import InitWindowValidateException
from ..base.top_window import TopWindow
from ..script import Script

# torchinfo: https://github.com/TylerYep/torchinfo

class ModelSummaryWindow(TopWindow):
    command_label = 'Model Summary'
    def __init__(self, parent, trainers):
        super().__init__(parent, 'Model summary')
        self.trainers = trainers
        self.trainer = None
        self.current_plot = None # plan name
        # self.plan_to_plot = None # plan name
        self.check_data()

        self.script_history = Script()
        self.script_history.add_import("from torchinfo import summary")

        self.summary = tk.Text(self, )
        self.summary.insert(tk.END, "")

        # init data
        ## fetch plan list
        trainer_map = {trainer.get_name(): trainer for trainer in trainers}
        trainer_list = ['Select a plan', *list(trainer_map.keys())]

        #+ gui
        ##+ option menu
        selector_frame = tk.Frame(self)
        ###+ plan
        selected_plan_name = tk.StringVar(self)
        selected_plan_name.set(trainer_list[0])
        selected_plan_name.trace_add('w', self.on_plan_select) # callback
        plan_opt = tk.OptionMenu(selector_frame, selected_plan_name, *trainer_list)


        plan_opt.pack()
        selector_frame.grid(row=0, column=0, sticky='news')
        self.summary.grid(row=1, column=0)

        self.selector_frame = selector_frame
        self.plan_opt = plan_opt
        self.trainer_map = trainer_map
        self.selected_plan_name = selected_plan_name

        self.drawCounter = 0
        self.update_loop()

    def check_data(self):
        if not isinstance(self.trainers, list) or len(self.trainers) == 0:
            raise InitWindowValidateException(
                self,
                'No valid training plan is generated'
            )

    def on_plan_select(self, var_name, *args):
        self.set_selection(False)
        # self.plan_to_plot = None
        self.trainer = None
        if self.getvar(var_name) not in self.trainer_map:
            return
        trainer = self.trainer_map[self.getvar(var_name)]
        if trainer is None:
            return
        self.trainer = trainer


    def update_loop(self):
        if (
            (self.trainer is not None) and
            (self.current_plot is None or self.current_plot != self.trainer)
        ):
                self.current_plot = self.trainer
                model_instance = self.trainer.model_holder.get_model(
                    self.trainer.dataset.get_epoch_data().get_model_args()
                ).to(self.trainer.option.get_device())
                X, _ = self.trainer.dataset.get_training_data()
                train_shape = (self.trainer.option.bs, 1, *X.shape[-2:])
                summary_object = summary(
                    model_instance, input_size = train_shape, verbose = 0
                )
                self.summary.delete("1.0", "end")
                self.summary.insert(tk.END, str(summary_object))


        counter = self.drawCounter
        if counter == self.drawCounter:
            self.set_selection(allow=True)

        self.after(100, self.update_loop)


        if self.selected_plan_name.get() not in self.trainer_map:
            return


    def set_selection(self, allow):
        state = None
        if not allow:
            self.drawCounter += 1
            if self.plan_opt['state'] != tk.DISABLED:
                state = tk.DISABLED
        elif self.plan_opt['state'] == tk.DISABLED:
                state = tk.NORMAL
        if state:
            self.plan_opt.config(state=state)
