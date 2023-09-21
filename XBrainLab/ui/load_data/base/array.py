from .array_setter import ArrayInfoSetter
from .base import LoadBase
from .info import RawInfo


class LoadArray(LoadBase):
    def __init__(self, parent, title):
        super().__init__(parent, title)

    def reset(self):
        self.raw_info = RawInfo()
        super().reset()

    def handle_array(self, filepath, selected_data):
        if not self.raw_info.is_info_complete():
            array_info_module = ArrayInfoSetter(
                self, filepath, selected_data, self.raw_info,
                type_ctrl=self.type_ctrl.get()
            )
            array_info = array_info_module.get_result()

            if not array_info:
                return False
            self.array_info = array_info

        data, generation_script = self.array_info.generate_mne(
            filepath, selected_data, self.type_ctrl.get()
        )
        self.script_history += generation_script
        return data
