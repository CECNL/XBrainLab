from copy import deepcopy
import mne
import os
class Raw:
    def __init__(self, filepath, mne_data):
        self.filepath = filepath
        self.mne_data = mne_data
        self.preprocess_history = []
        self.raw_events = None
        self.raw_event_id = None
        self.subject = 0
        self.session = 0
    
    def get_filepath(self):
        return self.filepath

    def get_filename(self):
        return os.path.basename(self.filepath)
    
    def get_subject_name(self):
        return str(self.subject)

    def get_session_name(self):
        return str(self.session)
    
    def get_preprocess_history(self):
        return self.preprocess_history
    
    def add_preprocess(self, desc):
        self.preprocess_history.append(desc)

    def set_subject_name(self, subject):
        self.subject = subject

    def set_session_name(self, session):
        self.session = session

    def set_event(self, events, event_id):
        if self.is_raw():
            self.raw_events = events
            self.raw_event_id = event_id
        else:
            assert self.get_epochs_length() == len(events)
            self.mne_data.events = events
            self.mne_data.event_id = event_id
    
    def set_mne(self, data):
        if isinstance(data,  mne.epochs.BaseEpochs):
            if self.raw_event_id:
                assert len(self.raw_events) == len(data.events)
                data.events = self.raw_events
                data.event_id = self.raw_event_id
                self.raw_events = None
                self.raw_event_id = None
        self.mne_data = data
    
    def set_mne_and_wipe_events(self, data):
        self.raw_events = None
        self.raw_event_id = None
        self.mne_data = data

    #
    def get_mne(self):
        return self.mne_data

    def get_nchan(self):
        return self.mne_data.info['nchan']

    def get_sfreq(self):
        return self.mne_data.info['sfreq']

    def get_filter_range(self):
        return self.mne_data.info['highpass'], self.mne_data.info['lowpass']
        
    def get_epochs_length(self):
        if self.is_raw():
            return 1
        return len(self.mne_data.events)
    
    def get_epoch_duration(self):
        return self.mne_data.get_data().shape[-1]

    def is_raw(self):
        return isinstance(self.mne_data, mne.io.base.BaseRaw)
    #
    def get_raw_event_list(self):
        try:
            if self.mne_data.event_id:
                return self.mne_data.events, self.mne_data.event_id
        except:
            pass
        try:
            return mne.find_events(self.mne_data)
        except:
            try:
                return mne.events_from_annotations(self.mne_data)
            except: 
                pass
        return None, None

    def get_event_list(self):
        if self.raw_event_id:
            return self.raw_events, self.raw_event_id
        return self.get_raw_event_list()

    def has_event(self):
        events, event_id = self.get_event_list()
        if event_id:
            return True
        return False

    def has_event_str(self):
        if self.has_event():
            return 'yes'
        return 'no'
    
    def get_event_name_list_str(self):
        events, event_id = self.get_event_list()
        if event_id:
            return ','.join([str(e) for e in event_id])
        return 'None'
    #
    def get_row_info(self):
        channel = self.get_nchan()
        sfreq = self.get_sfreq()
        epochs = self.get_epochs_length()
        has_event = self.has_event_str()
        return self.get_filename(), self.subject, self.session, channel, sfreq, epochs, has_event
