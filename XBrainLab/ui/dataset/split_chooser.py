import tkinter as tk

from ..base import TopWindow

class ManualSplitChooser(TopWindow):
    def __init__(self, parent, choices):
        super().__init__(parent, 'Manual')
        self.selected = []
        scrollbar = tk.Scrollbar(self)
        listbox = tk.Listbox(self, selectmode="extended", yscrollcommand=scrollbar.set)
        
        for idx, name in choices:
            listbox.insert(tk.END, name)
        scrollbar.config(command=listbox.yview)
        
        listbox.grid(row=1, column=0, padx=10, pady=10, sticky='news')
        scrollbar.grid(row=1, column=1, pady=10, sticky='news')
        tk.Button(self, text="Confirm", command=self.confirm, width=8).grid(
            row=2, column=0, columnspan=2
        )
        self.listbox = listbox
    
    def confirm(self):
        self.selected = list(self.listbox.curselection())
        self.destroy()
    
    def _get_result(self):
        return self.selected
    
