from .base import PreprocessBase
import scipy.io
import numpy as np
import os

class Export(PreprocessBase):

    def data_preprocess(self, filepath):
        for preprocessed_data in self.preprocessed_data_list:
            x = preprocessed_data.get_mne().get_data()
            if (preprocessed_data.has_event()):
                events, _ = preprocessed_data.get_event_list()
                y = events[:, -1]
            else:
                y = None
            history = preprocessed_data.get_preprocess_history()
        
            output = {}
            output['x'] = x
            if y is not None:
                output['y'] = y
            if history:
                output['history'] = history
            filename = 'Sub-' + preprocessed_data.get_subject_name()
            filename += '_'
            filename += 'Sess-' + preprocessed_data.get_session_name()
            filename += '.mat'
            scipy.io.savemat(os.path.join(filepath, filename), output)
            
        return self.preprocessed_data_list
