from .base import PreprocessBase
import numpy as np

class EditEventName(PreprocessBase):

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError(f"Event name can only be edited for epoched data")

    def get_preprocess_desc(self, new_event_name):
        diff = {k for k in new_event_name.keys()}.difference({v for v in new_event_name.values()})
        return f"Update {len(diff)} event names"

    def _data_preprocess(self, preprocessed_data, new_event_name):
        # update parent event name to event id dict
        assert [k for k in new_event_name.keys()] != [v for v in new_event_name.values()], "No Event name updated."
        if len([k for k in new_event_name.keys()]) != len([v for v in new_event_name.values()]):
            print("UserWarning: Updated with duplicate new event names.")
        
        events, event_id = preprocessed_data.get_event_list()
        new_event_id = {}
        for e in event_id:
            new_event_id[new_event_name[e]] = event_id[e]
        preprocessed_data.set_event(events, new_event_id)

class EditEventId(PreprocessBase):

    def check_data(self):
        super().check_data()
        for preprocessed_data in self.preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError(f"Event id can only be edited for epoched data")

    def get_preprocess_desc(self, new_event_ids):
        diff = {k for k in new_event_ids.keys()}.difference({v for v in new_event_ids.values()})
        return f"Update {len(diff)} event ids"

    def _data_preprocess(self, preprocessed_data, new_event_ids):
        # update parent event data
        assert [k for k in new_event_ids.keys()] != [v for v in new_event_ids.values()], "No Event Id updated."

        events, event_id = preprocessed_data.get_event_list()
        new_events, new_event_id = events.copy(), dict()
        if len(np.unique(new_event_ids.keys())) == len(np.unique(new_event_ids.values())):
            print("UserWarning: Updated with duplicate new event Ids. Event names of same event id are automatically merged.")
            uq, cnt = np.unique([v for v in new_event_ids.values()], return_counts=True) # [1,2,3], [1,1,2]
            dup = uq[cnt>1] # 3
            event_id_dup = {v:[] for v in dup} # 3: [768_2, 768_3]
            for k,v in event_id.items():
                if not new_event_ids[v] in dup:
                    new_event_id[k] = new_event_ids[v]
                else:
                    event_id_dup[new_event_ids[v]].append(k) # 

                new_events[np.where(events[:,-1]==v), -1] = new_event_ids[v]
            event_id_dup = {k:'/'.join(v) for k,v in event_id_dup.items()} #3: 768_2/768_3
            for k,v in event_id_dup.items():
                new_event_id[v] = k
        else:
            for k,v in event_id.items():
                new_event_id[k] = new_event_ids[v]
                new_events[np.where(events[:,-1]==v), -1] = new_event_ids[v]
       
        preprocessed_data.set_event(new_events, new_event_id)