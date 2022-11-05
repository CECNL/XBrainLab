import pytest
import tkinter as tk
from tkinter import filedialog
from XBrainLab.base import TopWindow
@pytest.fixture(scope="function", autouse=True)
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()

@pytest.fixture(scope="function", autouse=True)
def tk_warning():
    warning_list = []
    old_warning = tk.messagebox.showwarning
    tk.messagebox.showwarning = lambda **args: warning_list.append(args)
    yield warning_list
    tk.messagebox.showwarning = old_warning

@pytest.fixture(scope="function")
def mock_askopenfilenames():
    def get_askopenfilenames_generator(file_list):
        old_func = filedialog.askopenfilenames
        filedialog.askopenfilenames = lambda **args: file_list
        yield 1
        filedialog.askopenfilenames = old_func
    return get_askopenfilenames_generator
