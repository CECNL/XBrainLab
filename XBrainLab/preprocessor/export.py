import os

import scipy.io

from .base import PreprocessBase


class Export(PreprocessBase):
    """Preprocessing class for exporting data.

    Input:
        filepath: Path to save the data.
    """

    def data_preprocess(self, filepath: str):
        for preprocessed_data in self.preprocessed_data_list:
            # get X and y
            x = preprocessed_data.get_mne().get_data()
            if (preprocessed_data.has_event()):
                events, _ = preprocessed_data.get_event_list()
                y = events[:, -1]
            else:
                y = None
            # get history
            history = preprocessed_data.get_preprocess_history()

            # save data
            output = {}
            output['x'] = x
            if y is not None:
                output['y'] = y
            if history:
                output['history'] = history
            # prepare filename
            filename = 'Sub-' + preprocessed_data.get_subject_name() # subject
            filename += '_'
            filename += 'Sess-' + preprocessed_data.get_session_name() # session
            filename += '.mat'
            scipy.io.savemat(os.path.join(filepath, filename), output)

        return self.preprocessed_data_list
