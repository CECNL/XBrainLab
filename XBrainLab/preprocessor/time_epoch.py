from .base import PreprocessBase
import mne

class TimeEpoch(PreprocessBase):

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if not preprocessed_data.is_raw():
                raise ValueError(f"Only raw data can be epoched, got epochs")
            events, event_id = preprocessed_data.get_raw_event_list()
            if not event_id:
                raise ValueError(f"No event markers found for {preprocessed_data.get_filename()}")

    def get_preprocess_desc(self, baseline, new_event_id, tmin, tmax):
        return f"Epoching {tmin} ~ {tmax} by event"

    def _data_preprocess(self, preprocessed_data, baseline, new_event_id, tmin, tmax):
        events, event_id = preprocessed_data.get_raw_event_list()
        selected_event_id = {}
        for event_name in new_event_id:
            selected_event_id[ event_name ] = event_id[event_name]
        data = mne.Epochs(preprocessed_data.get_mne(), events,
                                        event_id=selected_event_id,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline,
                                        preload=True)
        try:
            preprocessed_data.set_mne(data)
        except:
            raise ValueError(f'Inconsistent number of events with label length (got {len(events)})')
