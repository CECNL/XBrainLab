from .base import PanelBase

class TrainingSchemePanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Scheme', **args)
