import tkinter as tk

from ..load_data import DataType
from .base import PanelBase


class DatasetPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Dataset', **args)
        frame = tk.Frame(self)

        tk.Label(frame, text='Type').grid(row=0, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Subject').grid(row=1, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Session').grid(row=2, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Epochs').grid(row=3, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Channel').grid(row=4, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Sample rate').grid(row=5, column=0, sticky='e', padx=10)
        tk.Label(frame, text='tmin (sec.)').grid(row=6, column=0, sticky='e', padx=10)
        tk.Label(frame, text='duration (sec.)').grid(
            row=7, column=0, sticky='e', padx=10
        )
        tk.Label(frame, text='Highpass').grid(row=8, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Lowpass').grid(row=9, column=0, sticky='e', padx=10)
        tk.Label(frame, text='Classes').grid(row=10, column=0, sticky='e', padx=10)

        self.type_label = tk.Label(frame, text='None')
        self.subject_label = tk.Label(frame, text='None')
        self.session_label = tk.Label(frame, text='None')
        self.epochs_label = tk.Label(frame, text='None')
        self.channel_label = tk.Label(frame, text='None')
        self.sfreq_label = tk.Label(frame, text='None')
        self.tmin_label = tk.Label(frame, text='None')
        self.duration_label = tk.Label(frame, text='None')
        self.highpass_label = tk.Label(frame, text='None')
        self.lowpass_label = tk.Label(frame, text='None')
        self.classes_label = tk.Label(frame, text='None')

        self.type_label.grid(row=0, column=1)
        self.subject_label.grid(row=1, column=1)
        self.session_label.grid(row=2, column=1)
        self.epochs_label.grid(row=3, column=1)
        self.channel_label.grid(row=4, column=1)
        self.sfreq_label.grid(row=5, column=1)
        self.tmin_label.grid(row=6, column=1)
        self.duration_label.grid(row=7, column=1)
        self.highpass_label.grid(row=8, column=1)
        self.lowpass_label.grid(row=9, column=1)
        self.classes_label.grid(row=10, column=1)
        frame.pack(expand=True)

    def reset(self):
        self.type_label.config(text='None')
        self.subject_label.config(text='None')
        self.session_label.config(text='None')
        self.epochs_label.config(text='None')
        self.channel_label.config(text='None')
        self.sfreq_label.config(text='None')
        self.tmin_label.config(text='None')
        self.duration_label.config(text='None')
        self.highpass_label.config(text='None')
        self.lowpass_label.config(text='None')
        self.classes_label.config(text='None')

    def update_panel(self, preprocessed_data_list):
        self.reset()
        if not preprocessed_data_list:
            return

        subject_set = set()
        session_set = set()
        classes_set = set()
        epoch_length = 0
        for preprocessed_data in preprocessed_data_list:
            subject_set.add(preprocessed_data.get_subject_name())
            session_set.add(preprocessed_data.get_session_name())
            _, event_id = preprocessed_data.get_event_list()
            if event_id:
                classes_set.update(event_id)
            epoch_length += preprocessed_data.get_epochs_length()
        tmin = None
        duration = None
        if not preprocessed_data.is_raw():
            tmin = preprocessed_data.get_tmin()
            duration = int(
                preprocessed_data.get_epoch_duration() *
                100 /
                preprocessed_data.get_sfreq()
            ) / 100
        highpass, lowpass = preprocessed_data.get_filter_range()
        if preprocessed_data.is_raw():
            text = DataType.RAW.value
        else:
            text = DataType.EPOCH.value
        self.type_label.config(text=text)
        self.subject_label.config(text=len(subject_set))
        self.session_label.config(text=len(session_set))
        self.epochs_label.config(text=epoch_length)
        self.channel_label.config(text=preprocessed_data.get_nchan())
        self.sfreq_label.config(text=preprocessed_data.get_sfreq())
        self.tmin_label.config(text=tmin)
        self.duration_label.config(text=duration)
        self.highpass_label.config(text=highpass)
        self.lowpass_label.config(text=lowpass)
        self.classes_label.config(text=len(classes_set))
