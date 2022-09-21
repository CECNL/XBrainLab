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
            if parm == 'win' and isinstance(value.default, tk.Toplevel):
                self.win = value.default

    def __call__(self, *args):
        try:
            if self.subst:
                args = self.subst(*args)
            return self.func(*args)
        except CustomException as e:
            e.handle_exception()
        except Exception as e:
            parent = None
            try:
                if self.win.winfo_exists():
                    parent = self.win
            except:
                pass
            tk.messagebox.showerror(parent=parent, title='Error', message=e)
            
tk.CallWrapper = Catcher
def run():
    dash_board = None
    try:
        from .dash_board import DashBoard
        dash_board = DashBoard()
        dash_board.mainloop()
        return dash_board
    except Exception as e:
        try:
            dash_board.destroy()
        except:
            pass
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror(parent=root, title='Error', message=e)
        root.destroy()

