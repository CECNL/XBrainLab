import numpy as np
from enum import Enum
from .option import SplitUnit

class Raw:
    """
    raw_attr: {fn: [subject, session]}
    raw_data: {fn: mne.io.Raw}
    raw_event: {fn: [labels, event_ids]}
    """
    def __init__(self, raw_attr, raw_data, raw_event):
        self.id_map = {} # {fn: subject/session/data list position id}
        self.event_id_map = {} # {fn: label list position id}
        self.subject = []
        self.session = []
        self.label = []
        self.data = []
        self.event_id = {}
        self._init_attr(raw_attr=raw_attr, raw_data=raw_data, raw_event=raw_event)
    
    def _init_attr(self, raw_attr, raw_data, raw_event):
        i = 0
        for fn in raw_attr.keys():
            self.id_map[fn] = i
            self.subject.append(raw_attr[fn][0])
            self.session.append(raw_attr[fn][1])
            self.data.append(raw_data[fn])
            if fn in raw_event.keys():
                self.event_id_map[fn] = i
                self.label.append(raw_event[fn][0])
                if self.event_id == {}:
                    self.event_id = raw_event[fn][1]
                else:
                    assert self.event_id == raw_event[fn][1], 'Event id inconsistent.'
            i += 1
    def copy(self):
        newRaw = Raw()
        newRaw.id_map = self.id_map.copy()
        newRaw.event_id_map = self.event_id_map.copy()
        newRaw.subject = self.subject.copy()
        newRaw.session = self.session.copy()
        newRaw.label = self.label.copy()
        newRaw.data = [r.copy() for r in self.data]
        newRaw.event_id = self.event_id.copy()
        return newRaw
            
    def inspect(self):
        for k,v in self.id_map.items():
            #print(k, self.subject[v], self.session[v])
            print(self.data[v])
            #print(len(self.label[v]))
        print(self.event_id)
        #print(self.label)
        #print(self.event_id_map)

class Epochs:
    def __init__(self, epoch_attr={}, epoch_data={}):
        self.sfreq = None
        self.subject_map = {} # index: S{subject idx}
        self.session_map = {} # index: unique session num
        self.label_map = {} # index: event idx

        # shape(n_data_loaded * len_epoch)
        self.subject = np.array([]) # ([subject] * len(epochs) for n data)
        self.session = np.array([]) # ([session] * len(epochs) for n data)
        self.label = np.array([]) # (len(event_label) for n data)
        self.idx = np.array([]) # range(len(epochs)) for n data

        self.data = [] # mne structure

        self.event_id = {} # difference with label map: keys are event name string

        # ===== ? initialize somewhere else?
        if epoch_attr != {} and epoch_data !={}:
            self._init_epochs(epoch_attr=epoch_attr, epoch_data=epoch_data)

    def _init_epochs(self, epoch_attr, epoch_data):
        i = 0
        for fn in epoch_attr.keys():
            epoch_len = len(epoch_data[fn])
            self.subject = np.concatenate((self.subject, np.array([epoch_attr[fn][0]] * epoch_len)))
            self.session = np.concatenate((self.session, np.array([epoch_attr[fn][1]] * epoch_len)))
            self.label   = np.concatenate((self.label,   epoch_data[fn].events[:,2]))
            self.idx     = np.concatenate((self.idx,     range(epoch_len))) # epoch len
            self.data.append(epoch_data[fn])
            if self.event_id=={}:
                self.event_id = epoch_data[fn].event_id
            else: 
                assert epoch_data[fn].event_id == self.event_id, 'Event Id inconsistent.'
            i += 1
        self.sfreq = self.data[0].info['sfreq']
        self.label_map   = {i:i for i in np.unique(self.label)}
        self.session_map = {i:i for i in np.unique(self.session)}
        self.subject_map = {i:f"S{i+1}" for i in np.unique(self.subject)}
    
    def copy(self):
        newEpochs = Epochs()
        newEpochs.sfreq = self.sfreq
        newEpochs.subject_map = self.subject_map.copy()
        newEpochs.session_map = self.session_map.copy()
        newEpochs.label_map = self.label_map.copy()
        newEpochs.subject = self.subject.copy()
        newEpochs.session = self.session.copy()
        newEpochs.label = self.label.copy()
        newEpochs.idx = self.idx.copy()
        newEpochs.data = [e.copy() for e in self.data]
        newEpochs.event_id = self.event_id.copy()
        return newEpochs



    def get_args(self):
        return  {'n_classes': len(self.event_id), # max(np.unique(self.label)) + 1,
                 'channels' : self.data[-1].info['nchan'], #self.data[-1].shape[-2],
                 'samples'  : self.data[-1].get_data().shape[-1], #self.data[-1].shape[-1],
                 'sfreq'    : self.sfreq }
    def get_data_length(self):
        return len(self.data[-1])

    def pick(self, target_type, target_type_map, mask, num, skip, is_ratio, ref_exclude):
        ret = mask & False
        if is_ratio:
            if ref_exclude is None:
                num *= len(np.unique( target_type[mask]) )
            else:
                num *= len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
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
                        target = target_type[filtered_mask]
                        if len(target) > 0:
                            pos = (mask & (target_type == target[-1]))
                            ret |= pos
                            mask &= np.logical_not(pos)
                            num -= 1
        
        return ret, mask

    def pick_subject(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        # return self.pick(self.subject, self.subject_map, mask, num, skip, is_ratio, ref_exclude)
        target_type = self.subject
        target_type_map = self.subject_map
        ret = mask & False
        if split_type == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = len( np.unique( target_type[mask]) )
            else:
                target = len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_type == SplitUnit.RATIO:
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
    
    def pick_session(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        # return self.pick(self.session, self.session_map, mask, num, skip, is_ratio, ref_exclude)
        target_type = self.session
        target_type_map = self.session_map
        ret = mask & False
        if split_type == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = len( np.unique( target_type[mask]) )
            else:
                target = len(np.unique( np.concatenate([target_type[mask], target_type[ref_exclude]]) ))
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_type == SplitUnit.RATIO:
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
    
    def pick_trail(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        ret = mask & False
        if not mask.any():
            return ret, mask
        if split_type == SplitUnit.KFOLD:
            if ref_exclude is None:
                target = sum(mask)
            else:
                target = sum(mask) + sum(ref_exclude)
            inc = target % num
            num = target // num
            if inc > group_idx:
                num += 1
        elif split_type == SplitUnit.RATIO:
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

    def pick_subject_by_idx(self, idx):
        return self.subject == idx
    def inspect(self):
        print(self.data[0].events[:,2])
        print(self.event_id)

class DataSet:
    SEQ = 0
    def __init__(self, data_holder):
        self.name = ''
        self.dataset_id = DataSet.SEQ
        DataSet.SEQ += 1
        self.data_holder = data_holder

        data_length = data_holder.get_data_length()
        self.remaining = np.ones(data_length, dtype=bool)

        self.train = np.zeros(data_length, dtype=bool)
        self.kept_training_session_list = []
        self.kept_training_subject_list = []
        self.val = np.zeros(data_length, dtype=bool)
        self.test = np.zeros(data_length, dtype=bool)
        self.is_selected = True

    def set_selection(self, select):
        self.is_selected = select

    def get_remaining(self):
        return self.remaining.copy()
    
    def set_name(self, name):
        self.name = name
    
    def get_name(self):
        return str(self.dataset_id) + '-' + self.name

        
    def pick_subject(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        return self.data_holder.pick_subject(mask, num, split_type, ref_exclude, group_idx)
    
    def pick_session(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        return self.data_holder.pick_session(mask, num, split_type, ref_exclude, group_idx)
    
    def pick_trail(self, mask, num, split_type, ref_exclude=None, group_idx=None):
        return self.data_holder.pick_trail(mask, num, split_type, ref_exclude, group_idx)
    
    def pick_subject_by_idx(self, idx):
        self.remaining = self.data_holder.pick_subject_by_idx(idx)

    def filter_by_subject_idx(self, mask, idx):
        return mask & self.data_holder.pick_subject_by_idx(idx)

    def kept_training_session(self, mask):
        self.kept_training_session_list = np.unique(self.data_holder.session[mask])

    def kept_training_subject(self, mask):
        self.kept_training_subject_list = np.unique(self.data_holder.subject[mask])
    
    def set_test(self, mask):
        self.test = mask.copy()
        self.remaining &= np.logical_not(mask)
        for kept_training_session in self.kept_training_session_list:
            target = self.data_holder.session == kept_training_session
            self.train |= (self.remaining & target)
            self.remaining &= np.logical_not(target)
        for kept_training_subject in self.kept_training_subject_list:
            target = self.data_holder.subject == kept_training_subject
            self.train |= (self.remaining & target)
            self.remaining &= np.logical_not(target)
    
    def set_val(self, mask):
        self.val = mask.copy()
        self.remaining &= np.logical_not(mask)

    def set_train(self):
        self.train |= self.remaining
        self.remaining &= False

    def discard(self, mask):
        self.remaining &= np.logical_not(mask)

    def get_args(self):
        return self.data_holder.get_args()

    def get_training_data(self):
        X = self.data_holder.data[self.train]
        y = self.data_holder.label[self.train]
        return X, y
    
    def get_val_data(self):
        X = self.data_holder.data[self.val]
        y = self.data_holder.label[self.val]
        return X, y

    def get_test_data(self):
        X = self.data_holder.data[self.test]
        y = self.data_holder.label[self.test]
        return X, y

    def preview(self):
        for idx, d in enumerate([self.train, self.val]):
            print(idx)
            for s in range(9):
                for i in range(2):
                    for j in range(4):
                        for k in range(72):
                            print(d[s * 2 * 4 * 72 + i * 4 * 72 + j * 72 + k], end='\t')
                        print()
                    print()
                print()
            print()
            print()
