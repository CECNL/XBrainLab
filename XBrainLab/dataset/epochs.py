import numpy as np
from enum import Enum
from copy import deepcopy
from .option import SplitUnit
from ..utils import validate_list_type
from ..load_data import Raw
class Epochs:
    def __init__(self, preprocessed_data_list):
        validate_list_type(preprocessed_data_list, Raw, 'preprocessed_data_list')
        for preprocessed_data in preprocessed_data_list:
            if preprocessed_data_list[0].is_raw():
                raise ValueError(f"Items of preprocessed_data_list must be {Raw.__module__}.Raw of type epoch.")
        
        self.sfreq = None
        # maps
        self.subject_map = {} # index: subject name
        self.session_map = {} # index: session name
        self.label_map = {}   # {int(event_id): 'description'}
        self.event_id = {}    # {'event_name': int(event_id)}
        #
        self.channel_map = []
        self.channel_position = None

        # 1D np array
        self.subject = []
        self.session = []
        self.label = []
        self.idx = []

        self.data = []

        # event_id
        for preprocessed_data in preprocessed_data_list:
            _, event_id = preprocessed_data.get_event_list()
            self.event_id.update(event_id)
        ## fix
        fixed_event_id  = {}
        for event_name in self.event_id:
            fixed_event_id[event_name] = len(fixed_event_id)
        ## update
        self.event_id = fixed_event_id
        for preprocessed_data in preprocessed_data_list:
            old_events, old_event_id = preprocessed_data.get_event_list()
            
            old_event_id = old_event_id.copy()
            
            events = old_events.copy()
            event_id = old_event_id.copy()
            old_labels = old_events[:,2].copy()

            for old_event_name, old_event_label in old_event_id.items():
                events[:,2][ old_labels == old_event_label ] = fixed_event_id[old_event_name]
                event_id[old_event_name] = fixed_event_id[old_event_name]
            preprocessed_data.set_event(events, event_id)

        # label map
        self.label_map = {}
        for event_name, event_label in self.event_id.items():
            self.label_map[event_label] = event_name
        
        # info
        map_subject = {}
        map_session = {}
        
        for preprocessed_data in preprocessed_data_list:
            data = preprocessed_data.get_mne()
            epoch_len = preprocessed_data.get_epochs_length()
            subject_name = preprocessed_data.get_subject_name()
            session_name = preprocessed_data.get_session_name()
            if subject_name not in map_subject:
                map_subject[subject_name] = len(map_subject)
            if session_name not in map_session:
                map_session[session_name] = len(map_session)
            subject_idx = map_subject[subject_name]
            session_idx = map_session[session_name]

            self.subject = np.concatenate((self.subject, [subject_idx] * epoch_len))
            self.session = np.concatenate((self.session, [session_idx] * epoch_len))
            self.label   = np.concatenate((self.label,   data.events[:,2]))
            self.idx     = np.concatenate((self.idx,     range(epoch_len)))
            if self.data == []:
                self.data = data.get_data()
            else:
                self.data = np.concatenate((self.data,    data.get_data()))
            self.sfreq = data.info['sfreq']
            self.channel_map = data.info.ch_names.copy()

        self.session_map = {map_session[i]:i for i in map_session}
        self.subject_map = {map_subject[i]:i for i in map_subject}        

    def copy(self):
        return deepcopy(self)
       
    # data splitting
    ## get list
    def get_subject_list(self):
        return self.subject

    def get_session_list(self):
        return self.session

    def get_label_list(self):
        return self.label

    ## get list by mask
    def get_subject_list_by_mask(self, mask):
        return self.subject[mask]
    
    def get_session_list_by_mask(self, mask):
        return self.session[mask]
    
    def get_label_list_by_mask(self, mask):
        return self.label[mask]

    def get_idx_list_by_mask(self, mask):
        return self.idx[mask]

    ## get mapping
    def get_subject_name(self, idx):
        return self.subject_map[idx]

    def get_session_name(self, idx):
        return self.session_map[idx]

    def get_label_name(self, idx):
        return self.label_map[idx]

    ## get map
    def get_subject_map(self):
        return self.subject_map
    
    def get_session_map(self):
        return self.session_map
    
    def get_label_map(self):
        return self.label_map

    ## misc getter
    def get_subject_index_list(self):
        return list(self.subject_map.keys())

    def pick_subject_mask_by_idx(self, idx):
        return self.subject == idx
    
    ## data info
    def get_data_length(self): # return n_epochs * n_Epochs
        return len(self.data)

    ## picker
    def convert_list_to_counter(self, list_map, target):
        unique_idx, counts = np.unique(target, return_counts=True)
        idx_count = {i: 0 for i in set(list_map.keys())}
        for idx, count in zip(unique_idx, counts):
            idx_count[idx] = count
        idx_count = list(idx_count.items())
        idx_count.sort(key=lambda i: -i[1])
        
        return idx_count

    def get_mask_target(self, mask):
        filter_preview_mask = {}
        unique_label_idx = np.unique(self.get_label_list())
        unique_subject_idx = np.unique(self.get_subject_list())
        unique_session_idx = np.unique(self.get_session_list())
        for label_idx in unique_label_idx:
            if label_idx not in filter_preview_mask:
                filter_preview_mask[label_idx] = {}
            for subject_idx in unique_subject_idx:
                if subject_idx not in filter_preview_mask[label_idx]:
                    filter_preview_mask[label_idx][subject_idx] = {}
                for session_idx in unique_session_idx:
                    if session_idx in filter_preview_mask[label_idx][subject_idx]:
                        continue
                    filter_mask = (self.label == label_idx) & (self.subject == subject_idx) & (self.session == session_idx)
                    target_filter_mask = filter_mask & mask 
                    filter_preview_mask[label_idx][subject_idx][session_idx] = [target_filter_mask, 0]
        return filter_preview_mask

    def get_filtered_mask_pair(self, filter_preview_mask):
        min_count = self.get_data_length()
        filtered_mask_pair = None
        for label_idx in filter_preview_mask:
            unique_subject_idx = filter_preview_mask[label_idx]
            for subject_idx in unique_subject_idx:
                unique_session_idx = unique_subject_idx[subject_idx]
                for session_idx in unique_session_idx:
                    if unique_session_idx[session_idx][0].any() and unique_session_idx[session_idx][1] < min_count:
                        min_count = unique_session_idx[session_idx][1]
                        filtered_mask_pair = unique_session_idx[session_idx]
        return filtered_mask_pair

    def update_mask_target(self, filter_preview_mask, pos):
        for label_idx in filter_preview_mask:
            unique_subject_idx = filter_preview_mask[label_idx]
            for subject_idx in unique_subject_idx:
                unique_session_idx = unique_subject_idx[subject_idx]
                for session_idx in unique_session_idx:
                    filtered_mask_pair = unique_session_idx[session_idx]
                    filtered_mask_pair[1] += sum(filtered_mask_pair[0] & pos)
                    filtered_mask_pair[0] &= np.logical_not(pos)
        return filter_preview_mask


    def get_real_num(self, target_type, value, split_unit, mask, clean_mask, group_idx):
        if clean_mask is None:
            target = len(np.unique( target_type[mask] ))
        else:
            target = len(np.unique( target_type[clean_mask] ))
        if split_unit == SplitUnit.KFOLD:
            inc = target % value
            num = target // value
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            num = value * target
        elif split_unit == SplitUnit.NUMBER:
            num = value
        else:
            raise NotImplementedError
        num = int(num)
        return num

    def pick(self, target_type, mask, clean_mask, value, split_unit, group_idx):
        num = self.get_real_num(target_type, value, split_unit, mask, clean_mask, group_idx)
        ret = mask & False
        filter_preview_mask = self.get_mask_target(mask)
        while num > 0:
            filtered_mask_pair = self.get_filtered_mask_pair(filter_preview_mask)
            if filtered_mask_pair is None:
                return ret, mask
            target = target_type[filtered_mask_pair[0]]
            if len(target) > 0:
                pos = (mask & (target_type == target[-1]))
                ret |= pos
                mask &= np.logical_not(pos)
                self.update_mask_target(filter_preview_mask, pos)
                filtered_mask_pair[0] &= np.logical_not(pos)
                filtered_mask_pair[1] += sum(pos)
                num -= 1
        return ret, mask

    def pick_manual(self, target_type, mask, value):
        ret = mask & False
        for v in value:
            pos = (mask & (target_type == v))
            ret |= pos
            mask &= np.logical_not(pos)
        return ret, mask

    def pick_subject(self, mask, clean_mask, value, split_unit, group_idx):
        target_type = self.get_subject_list()
        if split_unit == SplitUnit.MANUAL:
            return self.pick_manual(target_type, mask, value)
        else:
            return self.pick(target_type, mask, clean_mask, value, split_unit, group_idx)

    def pick_session(self, mask, clean_mask, value, split_unit, group_idx):
        target_type = self.get_session_list()
        if split_unit == SplitUnit.MANUAL:
            return self.pick_manual(target_type, mask, value)
        else:
            return self.pick(target_type, mask, clean_mask, value, split_unit, group_idx)
        
    def pick_trial(self, mask, clean_mask, value, split_unit, group_idx):
        ret = mask & False
        if split_unit == SplitUnit.MANUAL:
            ret[value] = True
            ret &= mask
            mask &= np.logical_not(ret)
            return ret, mask
        
        if clean_mask is None:
            target = sum(mask)
        else:
            target = sum(clean_mask)
        if split_unit == SplitUnit.KFOLD:
            inc = target % value
            num = target // value
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            num = value * target
        elif split_unit == SplitUnit.NUMBER:
            num = value
        else:
            raise NotImplementedError
        num = int(num)

        filter_preview_mask = self.get_mask_target(mask)
        while num > 0:
            filtered_mask_pair = self.get_filtered_mask_pair(filter_preview_mask)
            if filtered_mask_pair is None:
                return ret, mask
            pos = filtered_mask_pair[0].nonzero()[0][-1]
            if mask[pos]:
                ret[pos] = True
                mask[pos] = False
                filtered_mask_pair[0][pos] = False
                filtered_mask_pair[1] += 1
                num -= 1
        return ret, mask

    # train
    def get_model_args(self):
        return  {'n_classes': len(self.label_map),
                 'channels' : len(self.channel_map),
                 'samples'  : self.data.shape[-1],
                 'sfreq'    : self.sfreq }
    
    def get_data(self):
        return self.data

    #eval
    def get_label_number(self):
        return len(self.label_map)

    def get_channel_names(self):
        return self.channel_map
    
    def set_channels(self, chs, channel_position):
        self.channel_map = chs
        self.channel_position = channel_position

    def get_montage_position(self):
        return self.channel_position