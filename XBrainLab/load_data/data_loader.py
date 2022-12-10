from ..utils import validate_type, validate_list_type
from . import Raw

class RawDataLoader(list):
    def __init__(self, raw_data_list=[]):
        validate_list_type(raw_data_list, Raw, "raw_data_list")
        super().__init__(raw_data_list)
        if raw_data_list:
            self.validate()

    def get_loaded_raw(self, filepath):
        for raw_data in self:
            if filepath == raw_data.get_filepath():
                return raw_data
        return None

    def validate(self):
        for i in range(len(self)):
            raw_data = self[i]
            self.check_loaded_data_consistency(raw_data, idx=0)
            _, event_id = raw_data.get_event_list()
            if not event_id:
                raise ValueError(f"No label has been loaded for {raw_data.get_filename()}")
        return True

    def check_loaded_data_consistency(self, raw, idx=-1):
        validate_type(raw, Raw, 'raw')
        if not self:
            return True
        # check channel number
        if self[idx].get_nchan() != raw.get_nchan():
            raise ValueError(f'Dataset channel numbers inconsistent (got {raw.get_nchan()}).')
        # check sfreq
        if self[idx].get_sfreq() != raw.get_sfreq():
            raise ValueError(f'Dataset sample frequency inconsistent (got {raw.get_sfreq()}).')
        # check same data type
        if self[idx].is_raw() != raw.is_raw():
            raise ValueError(message=f'Dataset type inconsistent.')
        # check epoch trial size
        if not raw.is_raw():
            if self[idx].get_epoch_duration() != raw.get_epoch_duration():
                raise ValueError(f'Epoch duration inconsistent (got {raw.get_epoch_duration()}).')
        return True
    
    def append(self, raw):
        self.check_loaded_data_consistency(raw)
        super().append(raw)

    def apply(self, study):
        from .. import XBrainLab
        validate_type(study, XBrainLab, 'study')
        self.validate()
        study.set_loaded_data_list(self)