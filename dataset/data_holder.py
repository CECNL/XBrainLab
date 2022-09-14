import numpy as np
class Epochs():
    def __init__(self):
        self.subject_map = {}
        self.session_map = {}
        self.label_map = {}

        self.subject = []
        self.session = []
        self.label = []
        self.idx = []

        self.data = []
    
    def get_data_length(self):
        return len(self.data)

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

    def pick_subject(self, mask, num, skip, is_ratio, ref_exclude=None):
        # return self.pick(self.subject, self.subject_map, mask, num, skip, is_ratio, ref_exclude)
        target_type = self.subject
        target_type_map = self.subject_map
        ret = mask & False
        if is_ratio:
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
    
    def pick_session(self, mask, num, skip, is_ratio, ref_exclude=None):
        # return self.pick(self.session, self.session_map, mask, num, skip, is_ratio, ref_exclude)
        target_type = self.session
        target_type_map = self.session_map
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
    
    def pick_trail(self, mask, num, skip, is_ratio, ref_exclude=None):
        ret = mask & False
        if not mask.any():
            return ret, mask
        if is_ratio:
            if ref_exclude is None:
                num *= sum(mask)
            else:
                num *= (sum(mask) + sum(ref_exclude))
        num = int(num)
        while num > 0:
            for subject in list(self.subject_map.keys())[::-1]:
                for session in list(self.session_map.keys())[::-1]:
                    for label in list(self.label_map.keys())[::-1]:
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
    
class DataSet():
    def __init__(self, data_holder):
        self.name = ''
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
        
    def pick_subject(self, mask, num, skip, is_ratio, ref_exclude=None):
        return self.data_holder.pick_subject(mask, num, skip, is_ratio, ref_exclude)
    
    def pick_session(self, mask, num, skip, is_ratio, ref_exclude=None):
        return self.data_holder.pick_session(mask, num, skip, is_ratio, ref_exclude)
    
    def pick_trail(self, mask, num, skip, is_ratio, ref_exclude=None):
        return self.data_holder.pick_trail(mask, num, skip, is_ratio, ref_exclude)
    
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