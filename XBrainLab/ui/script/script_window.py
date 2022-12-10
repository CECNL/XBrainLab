import tkinter as tk
import tkinter.filedialog
from ..base import TopWindow
from enum import Enum

class ScriptType(Enum):
    CLI = 'cli'
    UI = 'ui'
    ALL = 'all'

class ScriptPreview(TopWindow):
    def __init__(self, parent, script, script_type):
        super().__init__(parent, 'Script Preview')
        txt_edit = tk.Text(self)
        
        if script_type == ScriptType.CLI:
            self.content = script.get_str()
        elif script_type == ScriptType.UI:
            self.content = script.get_ui_str()
        elif script_type == ScriptType.ALL:
            self.content = script.get_all_str()
        else:
            raise NotImplementedError
            
        txt_edit.insert(tk.END, self.content)
        txt_edit.pack(expand=True, fill=tk.BOTH)

        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label='Export', command=self.export)
        self.config(menu=menu)
        self.txt_edit = txt_edit
    
    def export(self):
        filepath = tk.filedialog.asksaveasfilename(parent=self, initialfile="script.py", filetypes = (("python files","*.py"),))
        if filepath:
            content = self.txt_edit.get("1.0", tk.END)
            with open(filepath, 'w') as f:
                f.write(content)
            tk.messagebox.showinfo(parent=self, title='Success', message='Done')


