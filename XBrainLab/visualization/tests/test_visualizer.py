from XBrainLab.visualization import SaliencyMapViz, SaliencyTopoMapViz
from XBrainLab.training.record import EvalRecord
from XBrainLab.load_data import Raw
from XBrainLab.dataset import Epochs

import numpy as np
import mne
import pytest
from matplotlib import pyplot as plt


epoch_duration = 3
n_trial = 3
fs = 5
subject_list = ['1', '2', '3']
session_list = ['1', '2']
ch_names=['O1', 'O2']

def get_preprocessed_data_list(n_class):
    event_id = {'c' + str(i): i for i in range(n_class)}
    events = np.zeros((n_class * n_trial, 3), dtype=int)
    events[:, 0] = np.arange(events.shape[0])
    events[:, 2] = np.arange(n_class).repeat(n_trial)   

    ch_types = 'eeg'
    
    result = []
    for subject in subject_list:
        for session in session_list:
            base = int(subject) * 100000 + int(session) * 1000
            info = mne.create_info(ch_names=ch_names,
                           sfreq=fs,
                           ch_types=ch_types)
            data = np.zeros((len(events), len(ch_names), epoch_duration * fs))
            for i in range(len(events)):
                data[i, :, :] = base + events[i, 0]
            epochs = mne.EpochsArray(data, info, events=events, 
                                     tmin=0, event_id=event_id)
            raw = Raw(f"test/sub-{subject}_ses-{session}.fif", epochs)
            raw.set_subject_name(subject)
            raw.set_session_name(session)
            result.append(raw)
    return result

@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("epochs, n_class", [
    (Epochs(get_preprocessed_data_list(2)), 2),
    (Epochs(get_preprocessed_data_list(3)), 3),
    (Epochs(get_preprocessed_data_list(4)), 4),
])
@pytest.mark.parametrize("visualizer", [SaliencyMapViz, SaliencyTopoMapViz])
@pytest.mark.parametrize("mask_out", [True, False])
def test_map(absolute, epochs, n_class, visualizer, mask_out):
    label = np.ones(10)
    output = np.ones((10, 2))
    gradient = {
        i: np.zeros((10, 2, 4)) for i in range(n_class)
    }
    if mask_out:
        gradient[0] = np.array([])
        n_class -= 1
    epochs.set_channels(ch_names, np.random.rand(len(ch_names), 3))
    eval_record = EvalRecord(label, output, gradient)
    visualizer = visualizer(eval_record, epochs)
    visualizer.get_plt(absolute)
    assert sum([len(i.images) for i in visualizer.fig.axes]) == n_class
    plt.close(visualizer.fig)