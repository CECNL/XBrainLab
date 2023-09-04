from typing import List
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
            tk.messagebox.showwarning(parent=self, title="Warning", message="File expected to contain only corresponding event data.")
        else:
            mat_key = mat_key[0]
        event_content = mat_content[mat_key].squeeze().astype(np.int32)
        if len(event_content.shape)>1:
            self.label_list = event_content
            return event_content[:,-1].squeeze().tolist()
        else:
            self.label_list = event_content
            return event_content.tolist()

    def create_event(self, new_event_name: List[str]) -> tuple:
        """Create event array and event id.

        Args:
            new_event_name: List of event names.

        Returns:
            Tuple of event array and event id.
        """
        if self.label_list is not None:
            for e in new_event_name:
                if not new_event_name[e].strip():
                    raise ValueError("event name cannot be empty")
            
            if len(self.label_list.shape)>1:
                event_id = {new_event_name[i]: i for i in np.unique(self.label_list[:,-1])}
                events = self.label_list
            else:
                event_id = {new_event_name[i]: i for i in list(set(self.label_list))}
                events = np.zeros((len(self.label_list), 3), dtype=np.int32)
                print('UserWarning: Event array created without onset timesample. Please proceed with caution if operating on raw data without annotations.')
                events[:,0] = range(len(self.label_list))
                events[:,-1] = self.label_list
                
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