from ..base import TopWindow
from ..script import Script

class PreprocessBase(TopWindow):
    def __init__(self, parent, title, preprocessor):
        super().__init__(parent, title)
        self.return_data = None
        preprocessor.check_data()
        self.preprocessor = preprocessor
        self.script_history = Script()
        self.ret_script_history = None
        self.script_history.add_import('from XBrainLab import preprocessor')

    def _get_result(self):
        return self.return_data

    def _get_script_history(self):
        return self.ret_script_history
