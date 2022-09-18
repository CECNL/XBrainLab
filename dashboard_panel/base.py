import tkinter as tk
class PanelBase(tk.LabelFrame):
    def __init__(self, parent, text, row, column, rowspan=1, columnspan=1):
        super().__init__(parent, text=text)
        self.parent = parent
        
        self.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky='wesn', padx=10, pady=10)

    def update_panel(self, *args):
        """Override this to update values."""
        if (len(self.winfo_children()) == 0):
            tk.Label(self, text='Put content here').pack()