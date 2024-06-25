from __future__ import annotations

import os
import re
import traceback
from enum import Enum

import mne
import numpy as np

from ..utils import validate_type


class FilenameGroupKey(Enum):
    """
    Utility class for parsing filename with regex.
    """
    SUBJECT = 'subject'
    SESSION = 'session'

class Raw:
    """Class for storing raw data.

    Holds data loaded from `mne`, preprocess history and event information.

    Attributes:
        filepath: str
            Filepath of the raw data.
        mne_data: :class:`mne.io.BaseRaw` | :class:`mne.BaseEpochs`
            Loaded data from MNE.
        preprocess_history: list[str]
            List of preprocess history.
        # list of
        raw_events: list[list[int]] | None
            Raw events. Same as `mne` format,
            (onset, immediately preceding sample, event_id).
        raw_event_id: dict[str, int] | None
            Raw event id. Same as `mne` format, {event_name: event_id}.
        subject: str
            Subject name.
        session: str
            Session name.
    """

    def __init__(self, filepath: str, mne_data: mne.io.BaseRaw | mne.BaseEpochs):
        validate_type(filepath, str, 'filepath')
        validate_type(mne_data, (mne.io.BaseRaw, mne.BaseEpochs), 'mne_data')
        self.filepath = filepath
        self.mne_data = mne_data
        self.preprocess_history = []
        self.raw_events = None
        self.raw_event_id = None
        self.subject = 0
        self.session = 0

    def get_filepath(self) -> str:
        """Return the filepath of the raw data."""
        return self.filepath

    def get_filename(self) -> str:
        """Return the filename of the raw data."""
        return os.path.basename(self.filepath)

    def get_subject_name(self) -> str:
        """Return the subject name of the raw data."""
        return str(self.subject)

    def get_session_name(self) -> str:
        """Return the session name of the raw data."""
        return str(self.session)

    def get_preprocess_history(self) -> list[str]:
        """Return the preprocess history of the raw data."""
        return self.preprocess_history

    def add_preprocess(self, desc: str) -> None:
        """Add preprocess description to the preprocess history."""
        self.preprocess_history.append(desc)

    def parse_filename(self, regex: str) -> None:
        """Extract and set data related information from the filename.

        Args:
            regex: Regex for parsing filename.
        """
        try:
            filepath = self.get_filepath()
            filename = os.path.basename(filepath)
            m = re.match(regex, filename)
            groupdict = m.groupdict()
            if FilenameGroupKey.SESSION.value in groupdict:
                self.set_session_name(groupdict[FilenameGroupKey.SESSION.value])
            if FilenameGroupKey.SUBJECT.value in groupdict:
                self.set_subject_name(groupdict[FilenameGroupKey.SUBJECT.value])
        except Exception:
            traceback.print_exc()
            pass

    def set_subject_name(self, subject: str) -> None:
        """Set the subject name of the raw data."""
        self.subject = subject

    def set_session_name(self, session: str) -> None:
        """Set the session name of the raw data."""
        self.session = session

    def set_event(self, events: list[list[int]], event_id: dict[str, int]) -> None:
        """Set the event of the raw data.

        Args:
            events: Raw events. Same as `mne` format,
                    (onset, immediately preceding sample, event_id).
            event_id: Raw event id. Same as `mne` format, {event_name: event_id}.
        """
        validate_type(events, np.ndarray, 'events')
        validate_type(event_id, dict, 'event_id')
        assert len(events.shape) == 2 and events.shape[1] == 3
        if not self.is_raw():
            assert self.get_epochs_length() == len(events)
            self.mne_data.events = events
            self.mne_data.event_id = event_id
        self.raw_events = events
        self.raw_event_id = event_id

    def set_mne(self, data: mne.io.BaseRaw | mne.BaseEpochs) -> None:
        """Set new mne data.

        Args:
            data: New mne data.
        """
        # set loaded event to new data
        if (
            isinstance(data, mne.epochs.BaseEpochs) and
            self.raw_event_id
        ):
            # check event consistency
            if len(self.raw_events) != len(data.events):
                print(
                    'UserWarning: Number of events from loaded label file and selected events for epoching are inconsistent.'
                    'Please proceed with caution.'
                )
            data.events = self.raw_events
            data.event_id = self.raw_event_id
            self.raw_events = None
            self.raw_event_id = None
        self.mne_data = data

    def set_mne_and_wipe_events(self, data: mne.io.BaseRaw | mne.BaseEpochs) -> None:
        """Set new mne data and wipe loaded event.

        Args:
            data: New mne data.
        """
        self.raw_events = None
        self.raw_event_id = None
        self.mne_data = data

    # mne related functions
    def get_mne(self) -> mne.io.BaseRaw | mne.BaseEpochs:
        """Return the loaded data from MNE."""
        return self.mne_data

    def get_tmin(self) -> float:
        """Return the tmin of :attr:`mne_data`."""
        if self.is_raw():
            return 0.0
        return self.mne_data.tmin

    def get_nchan(self) -> int:
        """Return the number of channels of :attr:`mne_data`."""
        return self.mne_data.info['nchan']

    def get_sfreq(self) -> float:
        """Return the sample frequency of :attr:`mne_data`."""
        return self.mne_data.info['sfreq']

    def get_filter_range(self) -> tuple[float, float]:
        """Return the filter range of :attr:`mne_data`."""
        return self.mne_data.info['highpass'], self.mne_data.info['lowpass']

    def get_epochs_length(self) -> int:
        """Return the number of epochs."""
        if self.is_raw():
            return 1
        return len(self.mne_data.events)

    def get_epoch_duration(self) -> int:
        """Return the duration of each epoch in samples."""
        return self.mne_data.get_data().shape[-1]

    def is_raw(self) -> bool:
        """Return whether the data is unsegmented raw data."""
        return isinstance(self.mne_data, mne.io.base.BaseRaw)

    # event related functions
    def get_raw_event_list(self) -> tuple[list[list[int]], dict[str, int]]:
        """Return the event list and event id of the raw data
           directly from the :attr:`mne_data`.

        Returns:
            (events, event_id)
        """
        # epoch data
        try:
            if self.mne_data.event_id:
                return self.mne_data.events, self.mne_data.event_id
        except Exception:
            pass
        # stim channel
        try:
            events = mne.find_events(self.mne_data)
            event_ids = {
                str(e): e for e in np.unique(events[:, -1])
            }
        except Exception:
            return mne.events_from_annotations(self.mne_data)
        else:
            return events, event_ids

    def get_event_list(self) -> tuple[list[list[int]], dict[str, int]]:
        """Return the event list and event id of the raw data.

        Returns:
            (events, event_id)
        """
        if self.raw_event_id:
            return self.raw_events, self.raw_event_id
        return self.get_raw_event_list()

    def has_event(self) -> bool:
        """Return whether the data has event."""
        _, event_id = self.get_event_list()
        if event_id:
            return True
        return False

    def has_event_str(self) -> str:
        """Return whether the data has event in string format."""
        if self.has_event():
            return 'yes'
        return 'no'

    def get_event_name_list_str(self) -> str:
        """Return the event name list in string format. Separated by comma."""
        if not self.has_event():
            return 'None'
        _, event_id = self.get_event_list()
        return ','.join([str(e) for e in event_id])

    # misc
    def get_row_info(self) -> tuple[str, str, str, int, float, int, str]:
        """Return the information of the raw data for displaying in the UI table.

        Returns: (
            Filename, subject name, session name, number of channels,
            sample frequency, number of epochs, whether the data has event
        )
        """
        channel = self.get_nchan()
        sfreq = self.get_sfreq()
        epochs = self.get_epochs_length()
        has_event = self.has_event_str()
        return (
            self.get_filename(),
            self.get_subject_name(),
            self.get_session_name(),
            channel, sfreq, epochs, has_event
        )

