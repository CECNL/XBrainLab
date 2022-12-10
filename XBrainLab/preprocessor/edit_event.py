from .base import PreprocessBase

class EditEvent(PreprocessBase):

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError(f"Event name can only be edited for epoched data")

    def get_preprocess_desc(self, new_event_name):
        return f"Update {len(new_event_name)} event"

    def _data_preprocess(self, preprocessed_data, new_event_name):
        # update parent event data
        
        events, event_id = preprocessed_data.get_event_list()
        new_event_id = {}
        for e in event_id:
            new_event_id[new_event_name[e]] = event_id[e]
        preprocessed_data.set_event(events, new_event_id)