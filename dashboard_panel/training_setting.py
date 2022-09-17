from .base import PanelBase

class TrainingSettingPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Training Setting', **args)
