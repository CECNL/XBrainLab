import tkinter as tk
import tkinter.filedialog
from ..base import TopWindow
import os
import numpy as np

class ModelOutputWindow(TopWindow):
    command_label = 'Export Model Output (csv)'
    def __init__(self, parent, training_plan_holders):
        super().__init__(parent, self.command_label)
        self.training_plan_holders = training_plan_holders
        if not self.check_data():
            return

        # init data
        ## fetch plan list
        training_plan_map = {plan_holder.get_name(): plan_holder for plan_holder in training_plan_holders}
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
        selected_plan_name.trace('w', lambda *args: selected_real_plan_name.set(real_plan_list[0])) # reset selection
        real_plan_opt = tk.OptionMenu(self, selected_real_plan_name, *real_plan_list)

        tk.Label(self, text='Plan Name: ').grid(row=0, column=0)
        plan_opt.grid(row=0, column=1)
        tk.Label(self, text='Repeat: ').grid(row=1, column=0)
        real_plan_opt.grid(row=1, column=1)
        tk.Button(self, text='Export', command=self.export).grid(row=2, column=0, columnspan=2)

        self.real_plan_opt = real_plan_opt
        self.training_plan_map = training_plan_map
        self.real_plan_map = []

        self.selected_plan_name = selected_plan_name
        self.selected_real_plan_name = selected_real_plan_name


    def check_data(self):
        if type(self.training_plan_holders) != list:
            self.withdraw()
            tk.messagebox.showerror(parent=self, title='Error', message='No valid training plan is generated')
            self.destroy()
            return False
        return True

    def on_plan_select(self, var_name, *args):
        item_count = self.real_plan_opt['menu'].index(tk.END)
        if item_count >= 1:
            self.real_plan_opt['menu'].delete(1, item_count)
        if self.getvar(var_name) not in self.training_plan_map:
            return
        plan_holder = self.training_plan_map[self.getvar(var_name)]
        if plan_holder is None:
            return
        
        self.real_plan_map = {plan.get_name(): plan for plan in plan_holder.get_plans()}
        for choice in self.real_plan_map:
            self.real_plan_opt['menu'].add_command(label=choice, command=lambda value=choice: self.selected_real_plan_name.set(value))


    def export(self):
        if self.selected_real_plan_name.get() not in self.real_plan_map:
            tk.messagebox.showerror(parent=self, title='Error', message='Please select a training plan')
            return
        real_plan = self.real_plan_map[self.selected_real_plan_name.get()]
        record = real_plan.get_eval_record()
        if not record:
            tk.messagebox.showerror(parent=self, title='Error', message='No evaluation record for this training plan')
            return
        plan_name = self.training_plan_map[self.selected_plan_name.get()].get_name()
        plan_name += '-'+real_plan.get_name()+'.csv'
        filename = tk.filedialog.asksaveasfilename(parent=self, initialfile=plan_name, filetypes = (("csv files","*.csv"),))
        if filename:
            data = np.c_[record.output, record.label, record.output.argmax(axis=1)]
            try:
                np.savetxt(filename, data, delimiter=',', newline='\n', header=f'{",".join([str(i) for i in range(record.output.shape[1])])},ground_truth,predict', comments='')
                tk.messagebox.showinfo(parent=self, title='Success', message='Done')
            except Exception as e:
                tk.messagebox.showerror(parent=self, title='Error', message=str(e))
