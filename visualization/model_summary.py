from tabnanny import verbose
from torchinfo import summary
import tkinter as tk
from ..base.top_window import TopWindow


# torchinfo: https://github.com/TylerYep/torchinfo

class ModelSummaryWindow(TopWindow):
    def __init__(self, parent, model_instance):
        super().__init__(parent, 'Model summary')
        summary_object = summary(model_instance,
            # depth=
            # col_names=
            # ...
            verbose = 0
        )
        summary_text = tk.Text(self)
        summary_text.pack()
        summary_text.insert(tk.END, str(summary_object))
