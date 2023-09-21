import tkinter as tk

from .base import PanelBase

MAX_VALUE_LEN = 10
class TrainingSchemePanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Scheme', **args)
        # training
        training_label = tk.Label(self)

        # val frame
        validation_frame = tk.LabelFrame(self, text ='Validation')
        validation_frame.grid_columnconfigure([0, 1], weight=1)

        # test frame
        testing_frame = tk.LabelFrame(self, text='Testing')
        testing_frame.grid_columnconfigure([0, 1], weight=1)

        self.training_label = training_label
        self.validation_frame = validation_frame
        self.testing_frame = testing_frame

    def show_instruction(self):
        self.clear_panel()
        tk.Label(self, text='TODO: show steps').pack(expand=True)

    def show_panel(self):
        self.clear_panel()
        training_label = self.training_label
        validation_frame = self.validation_frame
        testing_frame = self.testing_frame

        training_label.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)
        testing_frame.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)
        validation_frame.pack(side=tk.TOP, padx=20, fill=tk.X, expand=True)

        self.is_setup = True

    def update_panel(self, datasets):
        if not self.is_setup:
            self.show_panel()

        training_label = self.training_label
        validation_frame = self.validation_frame
        testing_frame = self.testing_frame
        # clear
        training_label.config(text='Not set')
        for child in validation_frame.winfo_children():
            child.destroy()
        for child in testing_frame.winfo_children():
            child.destroy()
        if not datasets:
            return

        config = datasets[0].config

        training_label.config(text=config.train_type.value)
        val_splitter_list, test_splitter_list = config.get_splitter_option()

        # val frame
        row = 0
        for val_splitter in val_splitter_list:
            if val_splitter.is_option:
                ## init widget
                val_split_by_label = tk.Label(validation_frame, text=val_splitter.text)
                val_split_type_label = tk.Label(
                    validation_frame, text=val_splitter.split_unit.value
                )
                value_var = val_splitter.value_var
                if len(value_var) > MAX_VALUE_LEN:
                    value_var = value_var[:MAX_VALUE_LEN] + '...'
                val_split_entry_label = tk.Label(validation_frame, text=value_var)
                ## pack
                val_split_by_label.grid(row=row + 0, column=0, columnspan=2)
                val_split_type_label.grid(row=row + 1, column=0)
                val_split_entry_label.grid(row=row + 1, column=1)
                row += 2
            else:
                tk.Label(validation_frame, text=val_splitter.text).grid(
                    row=row, column=0, columnspan=2, pady=5
                )

        # test frame
        row = 0
        if config.is_cross_validation:
            tk.Label(testing_frame, text='Cross Validation').grid(
                row=row, column=0, columnspan=2
            )
            row += 1
        idx = 0
        for test_splitter in test_splitter_list:
            if test_splitter.is_option:
                idx += 1
                ## init widget
                test_split_by_label = tk.Label(testing_frame, text=test_splitter.text)
                test_split_type_label = tk.Label(
                    testing_frame, text=test_splitter.split_unit.value
                )
                value_var = test_splitter.value_var
                if len(value_var) > MAX_VALUE_LEN:
                    value_var = value_var[:MAX_VALUE_LEN] + '...'
                test_split_entry_label = tk.Label(testing_frame, text=value_var)
                ## pack
                test_split_by_label.grid(row=row + 0, column=0, columnspan=2)
                test_split_type_label.grid(row=row + 1, column=0)
                test_split_entry_label.grid(row=row + 1, column=1)
                row += 2
            else:
                tk.Label(testing_frame, text=test_splitter.text).grid(
                    row=row, column=0, columnspan=2, pady=5, sticky='news'
                )
                row += 1
