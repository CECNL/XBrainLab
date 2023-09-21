import tkinter as tk
from tkinter import ttk


class EditableTreeView(ttk.Treeview):
    def __init__(
        self,
        parent,
        editableCols=None,
        datas=None,
        deletable=False,
        delete_callback=None,
        column_width=100,
        height=5,
        **key
    ) -> None:
        if editableCols is None:
            editableCols = []
        super().__init__(parent, height=height, **key)
        self.editableCols = editableCols
        self.datas = datas
        self.deletable = deletable
        self.delete_callback = delete_callback
        self.bind('<Double-Button-1>', self.__editTable, self)
        self.editingEntry = None
        if 'columns' in key:
            for c in key['columns']:
                self.column(c, anchor=tk.CENTER, width=column_width)
                self.heading(c, text=c, anchor=tk.CENTER)

        if deletable:
            self.column("#0", width=50, stretch=tk.NO)
            self.heading("#0", text="remove", anchor=tk.CENTER)
        else:
            self.column("#0", width=column_width, stretch=tk.NO)

    def clear_deletable(self):
        self.deletable = False
        self.column("#0", width=0, stretch=tk.NO)

    def clear_editable(self):
        self.editableCols = []

    def clear_rows(self):
        self.delete(*self.get_children())

    def __editTable(self, event):
        self.submitEntry(None)
        item = self.identify_row(event.y)
        col = self.identify_column(event.x)
        if self.identify_region(event.x, event.y) == 'cell':
            if col not in self.editableCols:
                return
            x, y, width, height = self.bbox(item, col)
            value = self.set(item, col)

            self.editingEntry = ttk.Entry(self, justify='center') # edition entry
            self.editingEntry.place(x=x, y=y, width=width, height=height,
                        anchor='nw')  # display entry on top of cell
            self.editingEntry.insert(0, value)  # put former value in entry
            self.editingEntry.select_range(0, len(value))
            self.editingEntry.setvar('cell', (item, col))
            self.editingEntry.bind('<Escape>', self.__destroyEntry)
            self.editingEntry.bind('<FocusOut>', self.submitEntry)
            self.editingEntry.bind('<Return>', self.submitEntry)
            self.editingEntry.focus_set()
        elif (
            self.identify_region(event.x, event.y) == 'tree'
            and self.deletable
            and self.item(item)['text'] != ''
        ):
            idx = self.find_child_pos(item)
            self.delete(item)
            if self.datas is not None:
                del self.datas[idx]
            if self.delete_callback is not None:
                self.delete_callback()

    def set_datas(self, datas):
        self.datas = datas

    def get_selected_index(self):
        if len(self.selection()) == 0:
            return None
        return self.get_children().index(self.selection()[0])

    def find_child_pos(self, iid):
        return self.get_children().index(iid)

    def submitEntry(self, event=None):
        """Change item value."""
        if self.editingEntry is not None:
            item, col = self.editingEntry.getvar('cell')
            self.set(item, col, self.editingEntry.get())
            self.__destroyEntry(None)

    def __destroyEntry(self, event):
        if self.editingEntry is not None:
            self.editingEntry.destroy()
            self.editingEntry = None
