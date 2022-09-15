import tkinter as tk
class TopWindow(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__()
        self.title(title)
        
    def _get_result(self):
        """Override this to return values."""
        return None
        
    def get_result(self):
        try:
            self.wait_window()
        except:
            pass
        
        return self._get_result()