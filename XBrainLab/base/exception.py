import tkinter as tk
import tkinter.messagebox

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def handle_exception(self):
        """Override this to handle exception."""
        pass

class InitWindowValidateException(CustomException):
    def __init__(self, window, message):
        super().__init__(message)
        self.window = window

    def handle_exception(self):
        if self.window.winfo_exists():
            self.window.destroy()
        tk.messagebox.showerror(parent=self.window.master, title='Error', message=self)

class ValidateException(CustomException):
    def __init__(self, window, message):
        super().__init__(message)
        self.window = window
        
    def handle_exception(self):
        try:
            if not self.window.winfo_exists():
                self.window = None
        except:
            self.window = None
        tk.messagebox.showerror(parent=self.window, title='Error', message=self)

