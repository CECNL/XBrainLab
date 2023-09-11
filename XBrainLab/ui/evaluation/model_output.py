import tkinter as tk
from ..base import TopWindow, InitWindowValidateException, ValidateException
from ..script import Script

class ModelOutputWindow(TopWindow):
    command_label = 'Export Model Output (csv)'
    def __init__(self, parent, trainers):
        super().__init__(parent, self.command_label)
        self.trainers = trainers
        self.check_data()
        self.geometry("400x125")
        self.fix_window_size()

        # init data
        ## fetch plan list
        training_plan_map = {trainer.get_name(): trainer for trainer in trainers}
        training_plan_list = ['Select a plan'] + list(training_plan_map.keys())
        real_plan_list = ['Select repeat']

        #+ gui
        ##+ option menu
        ###+ plan
        selected_plan_name = tk.StringVar(self)
        selected_plan_name.set(training_plan_list[0])
        selected_plan_name.trace('w', self.on_plan_select) # callback
        plan_opt = tk.OptionMenu(self, selected_plan_name, *training_plan_list)
        ###+ real plan
        selected_real_plan_name = tk.StringVar(self)
        selected_real_plan_name.set(real_plan_list[0])
        selected_plan_name.trace(
            'w', 
            lambda *args, win=self: selected_real_plan_name.set(real_plan_list[0])
        ) # reset selection
        real_plan_opt = tk.OptionMenu(self, selected_real_plan_name, *real_plan_list)

        tk.Label(self, text='Plan Name: ').grid(
            row=0, column=0, sticky='e', pady=(10,0)
        )
        plan_opt.grid(row=0, column=1, sticky='w')
        tk.Label(self, text='Repeat: ').grid(row=1, column=0, sticky='e')
        real_plan_opt.grid(row=1, column=1, sticky='w')
        tk.Button(self, text='Export', command=self.export).grid(
            row=2, column=0, columnspan=2
        )
        self.columnconfigure([0,1], weight=1)
        self.rowconfigure([2], weight=1)


        self.real_plan_opt = real_plan_opt
        self.training_plan_map = training_plan_map
        self.real_plan_map = []

        self.selected_plan_name = selected_plan_name
        self.selected_real_plan_name = selected_real_plan_name
        self.script_history = Script()

    def check_data(self):
        if not isinstance(self.trainers, list) or len(self.trainers) == 0:
            raise InitWindowValidateException(
                self, 'No valid training plan is generated'
            )

    def on_plan_select(self, var_name, *args):
        item_count = self.real_plan_opt['menu'].index(tk.END)
        if item_count >= 1:
            self.real_plan_opt['menu'].delete(1, item_count)
        if self.getvar(var_name) not in self.training_plan_map:
            return
        trainer = self.training_plan_map[self.getvar(var_name)]
        if trainer is None:
            return
        
        self.real_plan_map = {plan.get_name(): plan for plan in trainer.get_plans()}
        for choice in self.real_plan_map:
            self.real_plan_opt['menu'].add_command(
                label=choice, 
                command=lambda win=self, v=choice: win.selected_real_plan_name.set(v)
            )

    def export(self):
        if self.selected_real_plan_name.get() not in self.real_plan_map:
            raise ValidateException(
                window=self, message='Please select a training plan'
            )
        real_plan = self.real_plan_map[self.selected_real_plan_name.get()]
        record = real_plan.get_eval_record()
        if not record:
            raise ValidateException(
                window=self, 
                message='No evaluation record for this training plan'
            )
        
        plan_name = self.training_plan_map[self.selected_plan_name.get()].get_name()
        plan_name += '-'+real_plan.get_name()+'.csv'
        filename = tk.filedialog.asksaveasfilename(
            parent=self, 
            initialdir=real_plan.target_path, 
            initialfile=plan_name, 
            filetypes = (("csv files","*.csv"),)
        )
        if filename:
            record.export_csv(filename)
            self.script_history.add_cmd(f"filepath={repr(filename)}", newline=True)
            self.script_history.add_cmd(
                f"plan_name={repr(self.selected_plan_name.get())}"
            )
            self.script_history.add_cmd(
                f"real_plan_name={repr(self.selected_real_plan_name.get())}"
            )
            self.script_history.add_cmd(
                "study.export_output_csv(filepath, plan_name, real_plan_name)"
            )
            
            tk.messagebox.showinfo(parent=self, title='Success', message='Done')
    
    def _get_script_history(self):
        return self.script_history