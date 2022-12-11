import traceback
import tkinter as tk
from .base import CustomException
import inspect

class Catcher:
    def __init__(self, func, subst, widget):
        self.func = func 
        self.subst = subst
        self.widget = widget
        self.win = None
        if hasattr(func, '__self__'):
            if isinstance(self.func.__self__, tk.Toplevel):
                self.win = self.func.__self__
        for parm, value in inspect.signature(self.func).parameters.items():
            if (parm == 'win' or parm == 'window') and isinstance(value.default, tk.Toplevel):
                self.win = value.default

    def __call__(self, *args):
        try:
            if self.subst:
                args = self.subst(*args)
            return self.func(*args)
        except CustomException as e:
            e.handle_exception()
        except Exception as e:
            traceback.print_exc()
            parent = None
            try:
                if self.win.window_exist:
                    parent = self.win
            except:
                pass
            tk.messagebox.showerror(parent=parent, title='Error', message=e)
            
tk.CallWrapper = Catcher