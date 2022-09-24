import numpy as np
from enum import Enum
from ..option import SplitUnit
from copy import deepcopy

class Epochs:
    def __init__(self, epoch_attr={}, mne_data={}):
        self.epoch_attr = epoch_attr # {'filename': (subject, session)}
        self.mne_data = mne_data # {'filename': mne structure}
        
        #
        self.sfreq = None
        self.subject_map = {} # index: subject name
        self.session_map = {} # index: session name
        self.event_id = {}    # {'event_name': int(event_id)}
        self.label_map = {}   # {int(event_id): 'description'}
        self.channel_map = []
        self.channel_position = None

        # 1D np array
        self.subject = []
        self.session = []
        self.label = []
        self.idx = []

        self.data = []
        
        self.update()
    
    def fix_event_id(self):
        # fix
        fixed_event_id  = {}
        for event_name in self.event_id:
            fixed_event_id[event_name] = len(fixed_event_id)
        # update
        self.event_id = fixed_event_id
        for filename in self.mne_data:
            data = self.mne_data[filename]
            old_event_id = data.event_id.copy()
            old_events = data.events[:,2].copy()
            for old_event_name, old_event_label in old_event_id.items():
                data.events[:,2][ old_events == old_event_label ] = fixed_event_id[old_event_name]
                data.event_id[old_event_name] = fixed_event_id[old_event_name]
        self.label_map = {}
        for event_name, event_label in fixed_event_id.items():
            self.label_map[event_label] = event_name

    def check_data(self):
        event_id_labels = self.event_id.values()
        if min(event_id_labels) != 0 or (max(event_id_labels) + 1 != len(event_id_labels)):
            self.fix_event_id()
            event_id_labels = self.event_id.values()

    def copy(self):
        return Epochs(self.epoch_attr.copy(), deepcopy(self.mne_data))

    def reset(self):
        self.sfreq = None
        self.subject_map = {} # index: subject name
        self.session_map = {} # index: session name
        self.event_id = {}    # {'event_name': int(event_id)}
        self.label_map = {}   # {int(event_id): 'description'}
        self.channel_map = []
        self.channel_position = None

        # 1D np array
        self.subject = []
        self.session = []
        self.label = []
        self.idx = []

        self.data = []
        # init
        for filename in self.mne_data.keys():
            self.event_id.update(self.mne_data[filename].event_id)

    # make sure to call this on every preprocessing
    def update(self):
        self.reset()
        self.check_data()
        
        map_subject = {}
        map_session = {}
        map_label = {}
        
        for filename in self.epoch_attr.keys():
            epoch_len = len(self.mne_data[filename].events)
            subject_name, session_name = self.epoch_attr[filename]
            if subject_name not in map_subject:
                map_subject[subject_name] = len(map_subject)
            if session_name not in map_session:
                map_session[session_name] = len(map_session)
            subject_idx = map_subject[subject_name]
            session_idx = map_session[session_name]

            self.subject = np.concatenate((self.subject, [subject_idx] * epoch_len))
            self.session = np.concatenate((self.session, [session_idx] * epoch_len))
            self.label   = np.concatenate((self.label,   self.mne_data[filename].events[:,2]))
            self.idx     = np.concatenate((self.idx,     range(epoch_len)))
            if self.data == []:
                self.data = self.mne_data[filename].get_data()
            else:
                self.data    = np.concatenate((self.data,    self.mne_data[filename].get_data()))
            self.sfreq = self.mne_data[filename].info['sfreq']
            self.channel_map = self.mne_data[filename].info.ch_names.copy()

        self.session_map = {map_session[i]:i for i in map_session}
        self.subject_map = {map_subject[i]:i for i in map_subject}

        for event_name, event_label in self.event_id.items():
            self.label_map[event_label] = event_name
        
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
    def pick_subject(self, mask, num, split_unit, ref_exclude=None, group_idx=None):
        # return self.pick(self.subject, self.subject_map, mask, num, skip, is_ratio, ref_exclude)
        target_type = self.subject
        target_type_map = self.subject_map
        ret = mask & False
        if split_unit == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = len( np.unique( target_type[mask]) )
            else:
                target = len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            if ref_exclude is None:
                num *= len(np.unique( target_type[mask]) )
            else:
                num *= len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
        num = int(num)
        while num > 0:
            for label in list(self.label_map.keys())[::-1]:
                for subject in list(self.subject_map.keys())[::-1]:
                    if num == 0:
                        break
                    if not mask.any():
                        return ret, mask
                    filtered_mask = (self.label == label) & (self.subject == subject)
                    filtered_mask = filtered_mask & mask
                    target = target_type[filtered_mask]
                    if len(target) > 0:
                        pos = (mask & (target_type == target[-1]))
                        ret |= pos
                        mask &= np.logical_not(pos)
                        num -= 1
                        break

        return ret, mask
    
    def pick_session(self, mask, num, split_unit, ref_exclude=None, group_idx=None):
        target_type = self.get_session_list()
        target_type_map = self.get_session_map()
        ret = mask & False
        if split_unit == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = len( np.unique( target_type[mask]) )
            else:
                target = len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            if ref_exclude is None:
                num *= len(np.unique( target_type[mask]) )
            else:
                num *= len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
        num = int(num)
        while num > 0:
            for label in list(self.label_map.keys())[::-1]:
                for session in list(self.session_map.keys())[::-1]:
                    if num == 0:
                        break
                    if not mask.any():
                        return ret, mask
                    filtered_mask = (self.label == label) & (self.session == session)
                    filtered_mask = filtered_mask & mask
                    target = target_type[filtered_mask]
                    if len(target) > 0:
                        pos = (mask & (target_type == target[-1]))
                        ret |= pos
                        mask &= np.logical_not(pos)
                        num -= 1
                        break
        
        return ret, mask
    
    def pick_trail(self, mask, num, split_unit, ref_exclude=None, group_idx=None):
        ret = mask & False
        if not mask.any():
            return ret, mask
        if split_unit == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = sum(mask)
            else:
                target = sum(mask) + sum(ref_exclude)
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            if ref_exclude is None:
                num *= sum(mask)
            else:
                num *= (sum(mask) + sum(ref_exclude))
        num = int(num)
        while num > 0:
            for label in list(self.label_map.keys())[::-1]:
                for session in list(self.session_map.keys())[::-1]:
                    for subject in list(self.subject_map.keys())[::-1]:
                        if num == 0:
                            break
                        if not mask.any():
                            return ret, mask
                        filtered_mask = (self.label == label) & (self.session == session) & (self.subject == subject)
                        filtered_mask = filtered_mask & mask
                        if filtered_mask.any():
                            pos = filtered_mask.nonzero()[0][-1]
                            if mask[pos]:
                                ret[pos] = True
                                mask[pos] = False
                            num -= 1
        return ret, mask

    # train
    def get_args(self):
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