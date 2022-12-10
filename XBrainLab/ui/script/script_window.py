import tkinter as tk
from ..base import TopWindow

class ScriptPreview(TopWindow):
    def __init__(self, parent, script):
        super().__init__(parent, 'Script Preview')
        txt_edit = tk.Text(self)
        txt_edit.insert(tk.END, script.get_str())
        txt_edit.pack(expand=True, fill=tk.BOTH)
