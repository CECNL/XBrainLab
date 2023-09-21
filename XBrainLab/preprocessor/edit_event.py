from __future__ import annotations

import numpy as np

from ..load_data import Raw
from .base import PreprocessBase


class EditEventName(PreprocessBase):
    """Preprocessing class for editing event name.

    Input:
        new_event_name: Mapping of old event name to new event name.
    """

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError("Event name can only be edited for epoched data")

    def get_preprocess_desc(self, new_event_name: dict[str, str]):
        diff = set(new_event_name.keys()).difference(
            set(new_event_name.values())
        )
        return f"Update {len(diff)} event names"

    def _data_preprocess(self, preprocessed_data: Raw, new_event_name: dict[str, str]):
        # update parent event name to event id dict
        events, event_id = preprocessed_data.get_event_list()
        for k in new_event_name:
            assert k in event_id, "New event name not found in old event name."
        assert (
            list(new_event_name.keys()) != list(new_event_name.values())
        ), "No Event name updated."

        new_event_id = {}
        for e in event_id:

            new_name = new_event_name[e] if e in new_event_name else e
            if new_name in new_event_id:
                raise ValueError(f"Duplicate event name: {new_name}")
            new_event_id[new_name] = event_id[e]

        preprocessed_data.set_event(events, new_event_id)

class EditEventId(PreprocessBase):
    """Preprocessing class for editing event id.

    Input:
        new_event_ids: Dict of new event id.
    """

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError("Event id can only be edited for epoched data")

    def get_preprocess_desc(self, new_event_ids: dict[str, int]):
        diff = set(new_event_ids.keys()).difference(
            set(new_event_ids.values())
        )
        return f"Update {len(diff)} event ids"

    def _data_preprocess(self, preprocessed_data: Raw, new_event_ids: dict[str, int]):
        # update parent event data
        assert (
            list(new_event_ids.keys()) != list(new_event_ids.values())
        ), "No Event Id updated."

        events, event_id = preprocessed_data.get_event_list()
        new_events, new_event_id = events.copy(), {}
        if (
            len(np.unique(new_event_ids.keys())) ==
            len(np.unique(new_event_ids.values()))
        ):
            print(
                "UserWarning: Updated with duplicate new event Ids. "
                "Event names of same event id are automatically merged."
            )
            uq, cnt = np.unique(
                list(new_event_ids.values()),
                return_counts=True
            ) # [1,2,3], [1,1,2]
            dup = uq[cnt>1] # 3
            event_id_dup = {v: [] for v in dup} # 3: [768_2, 768_3]
            for k, v in event_id.items():
                if new_event_ids[v] not in dup:
                    new_event_id[k] = new_event_ids[v]
                else:
                    event_id_dup[new_event_ids[v]].append(k) #

                new_events[np.where(events[:, -1]==v), -1] = new_event_ids[v]
            event_id_dup = {
                k: '/'.join(v)
                for k, v in event_id_dup.items()
            } #3: 768_2/768_3
            for k, v in event_id_dup.items():
                new_event_id[v] = k
        else:
            for k, v in event_id.items():
                new_event_id[k] = new_event_ids[v]
                new_events[np.where(events[:, -1]==v), -1] = new_event_ids[v]

        preprocessed_data.set_event(new_events, new_event_id)
