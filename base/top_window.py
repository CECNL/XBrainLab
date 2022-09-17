import tkinter as tk

class TopWindow(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__()
        self.parent = parent
        self.title(title)
        # recycle
        self.child_list = []
        try:
            parent.append_child_window(self)
        except:
            pass
        # put window
        toplevel_offsetx, toplevel_offsety = parent.winfo_x(), parent.winfo_y()
        padx = 25
        pady = 20
        self.geometry(f"+{toplevel_offsetx + padx}+{toplevel_offsety + pady}")

    def append_child_window(self, child):
        self.child_list.append(child)
    
    def remove_child_window(self, child):
        self.child_list.remove(child)
        
    def _get_result(self):
        """Override this to return values."""
        return None
        
    def get_result(self):
        try:
            self.wait_window()
        except:
            pass
        
        return self._get_result()

    def destroy(self):
        # close all children window
        child_list = self.child_list.copy()
        for child in child_list:
            if child.destroy():
                return True
        # remove self from parent
        try:
            self.parent.remove_child_window(self)
        except:
            pass
        super().destroy()