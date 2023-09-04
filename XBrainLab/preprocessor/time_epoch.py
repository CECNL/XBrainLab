from .base import PreprocessBase
from ..load_data import Raw
import numpy as np
import mne

class TimeEpoch(PreprocessBase):
    """Class for epoching data by event markers
    
    Input:
        baseline: Baseline removal time interval
        new_event_id: List of event names to be kept
        tmin: Start time before event marker
        tmax: End time after event marker
    """

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if not preprocessed_data.is_raw():
                raise ValueError(f"Only raw data can be epoched, got epochs")
            _, event_id = preprocessed_data.get_raw_event_list()
            if not event_id:
                raise ValueError(f"No event markers found for {preprocessed_data.get_filename()}")

    def get_preprocess_desc(self, baseline: list, new_event_id: dict, tmin: float, tmax: float):
        return f"Epoching {tmin} ~ {tmax} by event"

    def _data_preprocess(self, preprocessed_data: Raw, baseline: list, new_event_id: dict, tmin: float, tmax: float):
        events, event_id = preprocessed_data.get_raw_event_list()
        selected_event_id = {}
        selected_events = events.copy()
        for event_name in new_event_id:
            if event_name in event_id.keys():
                selected_event_id[event_name] = event_id[event_name]
        
        for iter in np.unique(events[:,-1]):
            if iter not in selected_event_id.values():
                selected_events = np.delete(selected_events, np.where(selected_events[:,-1]==iter)[0], 0)

        preprocessed_data.raw_event_id = selected_event_id
        preprocessed_data.raw_events = selected_events

        data = mne.Epochs(preprocessed_data.get_mne(), selected_events,
                                        event_id=selected_event_id,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline,
                                        preload=True)
        try:
            preprocessed_data.set_mne(data)
        except:
            raise ValueError(f'Inconsistent number of events with label length (got {len(events)})')
