from .base import PreprocessBase
import mne

class WindowEpoch(PreprocessBase):

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if not preprocessed_data.is_raw():
                raise ValueError(f"Only raw data can be epoched, got epochs")
            events, event_id = preprocessed_data.get_event_list()
            if not event_id:
                raise ValueError(f"No event markers found for {preprocessed_data.get_filename()}")
            if len(events) != 1 or len(event_id) != 1:
                raise ValueError(f"Should only contain single event label, found events={len(events)}, event_id={len(event_id)}")

    def get_preprocess_desc(self, duration, overlap):
        return f"Epoching {duration}s ({overlap}s overlap) by sliding window"

    def _data_preprocess(self, preprocessed_data, duration, overlap):
        mne_data = preprocessed_data.get_mne()
        duration = float(duration)
        overlap = 0.0 if overlap == "" else float(overlap)
        FIXED_ID = 0
        epoch = mne.make_fixed_length_epochs(mne_data, duration=duration, overlap=overlap, preload=True, id=FIXED_ID)
        _, event_id = preprocessed_data.get_event_list()
        epoch.event_id = {list(event_id.keys())[0]: FIXED_ID}
        try:
            preprocessed_data.set_mne_and_wipe_events(epoch)
        except:
            raise ValueError(f'Inconsistent number of events with label length.')