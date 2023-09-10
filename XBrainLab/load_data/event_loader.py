from __future__ import annotations
import numpy as np
import scipy.io
import tkinter as tk

from . import Raw
from ..utils import validate_type

class EventLoader:
    """Helper class for loading event data.

    Attributes:
        raw: :class:`Raw` 
            Raw data.
        label_list: List[int] | None
            List of event codes.
        events: list[list[int]] | None
            Event array. Same as `mne` format.
        event_id: dict[str, int] | None
            Event id. Same as `mne` format.
    """

    def __init__(self, raw: Raw):
        validate_type(raw, Raw, 'raw')
        self.raw = raw
        self.label_list = None
        self.events = None
        self.event_id = None

    def read_txt(self, selected_file: str) -> list:
        """Read event data from txt file.

        The txt file should contain a list of event codes, separated by space.

        Args:
            selected_file: Path to the txt file.

        Returns:
            List of event codes.
        """
        label_list = []
        with open(selected_file, encoding='utf-8', mode='r') as fp:
            for line in fp.readlines():
                label_list += [int(l.rstrip()) for l in line.split(' ')] # for both (n,1) and (1,n) of labels
            fp.close()
        self.label_list = label_list
        return label_list

    def read_mat(self, selected_file: str) -> list:
        """Read event data from mat file.

        The mat file should contain exactly one variable, which is a list of event codes.

        Args:
            selected_file: Path to the mat file.

        Returns:
            List of event codes.
        """
        mat_content = scipy.io.loadmat(selected_file)
        mat_key = [k for k in mat_content.keys() if not k.startswith('_')]
        if len(mat_key) > 1:
            raise ValueError("Mat file should contain exactly one variable.")
        else:
            mat_key = mat_key[0]
        event_content = mat_content[mat_key].astype(np.int32)
        # for (n,1) and (1,n) of labels
        if len(event_content.shape) == 2:
            if event_content.shape[0] == 1:
                event_content = event_content[0]
            elif event_content.shape[1] == 1:
                event_content = event_content[:,0]
        # (n, 3)
        if len(event_content.shape) == 2:
            assert event_content.shape[1] == 3, "Event array should have 3 columns."
            self.label_list = event_content
            return event_content[:, -1].tolist()
        # (n,)
        elif len(event_content.shape) == 1:
            self.label_list = event_content
            return event_content.tolist()
        else:
            raise ValueError("Either 1d or 2d array is expected.")

    def create_event(self, event_name_map: dict[int, str]) -> tuple:
        """Create event array and event id.

        Args:
            event_name_map: Mapping from event code to event name.

        Returns:
            Tuple of event array and event id.
        """
        if self.label_list is not None and len(self.label_list) > 0:
            # check if new event name is valid
            for e in event_name_map:
                if not event_name_map[e].strip():
                    raise ValueError("Event name cannot be empty.")
                
            self.label_list = np.array(self.label_list)
            # label_list in (n,3) format
            if len(self.label_list.shape) > 1:
                # get new event id mapping
                event_id = {event_name_map[i]: i for i in np.unique(self.label_list[:,-1])}
                events = self.label_list
            # label_list in (n,) format
            else:
                # get new event id mapping
                event_id = {event_name_map[i]: i for i in np.unique(self.label_list)}
                # create new event array
                events = np.zeros((len(self.label_list), 3))
                events[:,0] = range(len(self.label_list))
                events[:,-1] = self.label_list
                print('UserWarning: Event array created without onset timesample. Please proceed with caution if operating on raw data without annotations.')

            # check if event array is consistent with raw data
            if not self.raw.is_raw():
                if self.raw.get_epochs_length() != len(events):
                    raise ValueError(f'Inconsistent number of events (got {len(events)})')
            self.events = events
            self.event_id = event_id
            return events, event_id
        else:
            raise ValueError("No label has been loaded.")

    def apply(self) -> None:
        """Apply the loaded event data to the raw data.

        Raises:
            ValueError: If no label has been loaded.
        """
        assert self.events is not None, "No label has been loaded."
        assert self.event_id is not None, "No label has been loaded."
        self.raw.set_event(self.events, self.event_id)