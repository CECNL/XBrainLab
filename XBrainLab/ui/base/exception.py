import tkinter as tk
import tkinter.messagebox


class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)

    def handle_exception(self):
        """Override this to handle exception."""
        raise NotImplementedError()

class InitWindowValidateException(CustomException):
    def __init__(self, window, message):
        super().__init__(message)
        self.window = window

    def handle_exception(self):
        if self.window.window_exist:
            self.window.destroy()
        tk.messagebox.showerror(parent=self.window.master, title='Error', message=self)

class ValidateException(CustomException):
    def __init__(self, window, message):
        super().__init__(message)
        self.window = window

    def handle_exception(self):
        try:
            if not self.window.window_exist:
                self.window = None
        except AttributeError:
            self.window = None
        tk.messagebox.showerror(parent=self.window, title='Error', message=self)

