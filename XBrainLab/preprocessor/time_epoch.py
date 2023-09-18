from .base import PreprocessBase
from ..load_data import Raw
import numpy as np
import mne

class TimeEpoch(PreprocessBase):
    """Class for epoching data by event markers
    
    Input:
        baseline: Baseline removal time interval
        selected_event_names: List of event names to be kept
        tmin: Start time before event marker
        tmax: End time after event marker
    """

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if not preprocessed_data.is_raw():
                raise ValueError("Only raw data can be epoched, got epochs")
            _, event_id = preprocessed_data.get_event_list()
            if not event_id:
                raise ValueError(
                    f"No event markers found for {preprocessed_data.get_filename()}"
                )

    def get_preprocess_desc(
        self, 
        baseline: list, 
        selected_event_names: list, 
        tmin: float, 
        tmax: float
    ):
        return f"Epoching {tmin} ~ {tmax} by event ({baseline} baseline)"

    def _data_preprocess(
        self, 
        preprocessed_data: Raw, 
        baseline: list, 
        selected_event_names: list, 
        tmin: float, 
        tmax: float
    ):
        raw_events, raw_event_id = preprocessed_data.get_raw_event_list()
        if(len(raw_events) == 0):
            raw_events, raw_event_id = preprocessed_data.get_event_list()
        selected_event_id = {}
        for event_name in selected_event_names:
            if event_name in raw_event_id.keys():
                selected_event_id[event_name] = raw_event_id[event_name]
        
        selection_mask = np.zeros(raw_events.shape[0], dtype=bool)
        for event_name in selected_event_id.keys():
            selection_mask = np.logical_or(
                selection_mask, raw_events[:,-1]==selected_event_id[event_name]
            )
        selected_events = raw_events[selection_mask]
        if(len(selected_events) == 0):
            raise ValueError("No event markers found.")

        data = mne.Epochs(preprocessed_data.get_mne(), selected_events,
                                        event_id=selected_event_id,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=baseline,
                                        preload=True)
        
        preprocessed_data.set_mne(data)