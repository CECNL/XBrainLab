from .base import PanelBase

class TrainingStatusPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Status', **args)
