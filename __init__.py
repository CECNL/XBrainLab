import tkinter as tk
import traceback
from .base import CustomException

class Catcher:
    def __init__(self, func, subst, widget):
        self.func = func 
        self.subst = subst
        self.widget = widget
    def __call__(self, *args):
        try:
            if self.subst:
                args = self.subst(*args)
            return self.func(*args)
        except Exception as e:
            if isinstance(e, CustomException):
                e.handle_exception()
                return
            traceback.print_exc()
            
tk.CallWrapper = Catcher
from .dash_board import DashBoard