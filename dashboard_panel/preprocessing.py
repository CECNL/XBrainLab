from .base import PanelBase

class PreprocessPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Preprocessing', **args)
