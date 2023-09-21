import tkinter as tk


class PanelBase(tk.LabelFrame):
    def __init__(self, parent, text, row, column, rowspan=1, columnspan=1):
        super().__init__(parent, text=text)
        self.parent = parent
        self.is_setup = False

        self.grid(
            row=row, column=column, rowspan=rowspan,
            columnspan=columnspan, sticky='wesn', padx=10, pady=10
        )

    def clear_panel(self):
        for child in self.winfo_children():
            child.grid_forget()
            child.pack_forget()
            child.forget()
        self.is_setup = False


    def update_panel(self, *args):
        """Override this to update values."""
        if (len(self.winfo_children()) == 0):
            tk.Label(self, text='Put content here').pack(expand=True)
