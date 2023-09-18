from XBrainLab.preprocessor.base import PreprocessBase
from XBrainLab.load_data import Raw

import mne
import pytest
import numpy as np

base_fs = 500
base_duration = 10

def _generate_mne(fs, ch_names, ch_types, length = base_duration):
    info = mne.create_info(ch_names=ch_names,
                           sfreq=fs,
                           ch_types=ch_types)
    data = np.random.RandomState(0).randn(len(ch_names), fs * length)
    return mne.io.RawArray(data, info)

# raw without event
@pytest.fixture
def raw():
    mne_raw = _generate_mne(base_fs, ['Fp1', 'Fp2', 'F3', 'F4'], 'eeg')
    return Raw('tests/test_data/sub-01_ses-01_task-rest_eeg.fif', mne_raw)


def test_base(raw):
    with pytest.raises(ValueError):
        PreprocessBase([])

    base = PreprocessBase([raw])
    assert len(base.get_preprocessed_data_list()) == 1

    with pytest.raises(NotImplementedError):
        base.get_preprocess_desc()
    
    with pytest.raises(NotImplementedError):
        base._data_preprocess()


def test_inherit(raw):
    class InheritedPreprocessor(PreprocessBase):
        def get_preprocess_desc(self, *args, **kargs):
            return "test desc " + str(args[0])

        def _data_preprocess(self, preprocessed_data, *args, **kargs):
            preprocessed_data.set_subject_name("test_inherit")
    
    preprocessor = InheritedPreprocessor([raw])
    preprocessor.data_preprocess(1)

    result = preprocessor.get_preprocessed_data_list()[0]

    assert result.get_subject_name() == "test_inherit"
    assert result.get_preprocess_history() == ["test desc 1"]