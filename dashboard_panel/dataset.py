from .base import PanelBase

class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)
