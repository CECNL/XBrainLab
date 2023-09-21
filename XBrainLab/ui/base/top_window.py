import contextlib
import tkinter as tk


class TopWindow(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__()
        self.parent = parent
        self.title(title)
        # recycle
        self.window_exist = True
        self.child_list = []
        with contextlib.suppress(Exception):
            parent.append_child_window(self)

        # put window
        toplevel_offsetx, toplevel_offsety = parent.winfo_x(), parent.winfo_y()
        padx = 25
        pady = 20
        self.geometry(f"+{toplevel_offsetx + padx}+{toplevel_offsety + pady}")
    #
    def append_child_window(self, child):
        self.child_list.append(child)

    def remove_child_window(self, child):
        self.child_list.remove(child)

    def fix_window_size(self):
        self.resizable(False, False)

    #
    def _get_result(self):
        """Override this to return values."""
        return None

    def _get_script_history(self):
        """Override this to return values."""
        return None

    def get_result(self):
        with contextlib.suppress(Exception):
            self.wait_window()


        return self._get_result()

    def get_script_history(self):
        with contextlib.suppress(Exception):
            self.wait_window()


        return self._get_script_history()

    #
    def destroy(self, force=False):
        # close all children window
        child_list = self.child_list.copy()
        for child in child_list:
            if not child.destroy(force):
                return False
        # remove self from parent
        with contextlib.suppress(Exception):
            self.parent.remove_child_window(self)

        self.window_exist = False
        super().destroy()
        return True
