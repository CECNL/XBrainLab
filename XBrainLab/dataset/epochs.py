from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np

from ..load_data import Raw
from ..utils import validate_list_type
from .option import SplitUnit


class TrialSelectionSequence(Enum):
    """Utility class for trial selection sequence in dataset splitting."""
    SESSION = 'session'
    SUBJECT = 'subject'
    Label = 'label'

class Epochs:
    """Class for storing epoch data.

    Handles list of `Raw` objects, which are preprocessed data,
        and convert them into epoch data.
    With functions to corporate with dataset generator.

    Parameters:
        preprocessed_data_list: List of preprocessed data.

    Attributes:
        sfreq: float
            Sampling frequency of the data.
        subject_map: dict[int, str]
            Mapping from subject index to subject name.
        session_map: dict[int, str]
            Mapping from session index to session name.
        label_map: dict[int, str]
            Mapping from label index to label name.
        event_id: dict[str, int]
            Mapping from event name to event id.
        ch_names: list[str]
            List of channel names.
        channel_position: list | None
            List of channel positions. None if not set.
            Channel format is (x, y, z).
        subject: np.ndarray
            List of subject index of each epoch.
        session: np.ndarray
            List of session index of each epoch.
        label: np.ndarray
            List of label index of each epoch.
        idx: np.ndarray
            List of epoch index within each preprocessed data.
    """
    def __init__(self, preprocessed_data_list: list[Raw]):
        validate_list_type(
            instance_list=preprocessed_data_list,
            type_class=Raw,
            message_name='preprocessed_data_list'
        )
        for preprocessed_data in preprocessed_data_list:
            if preprocessed_data.is_raw():
                raise ValueError(
                    "Items of preprocessed_data_list must be "
                    f"{Raw.__module__}.Raw of type epoch."
                )

        self.sfreq = None
        # maps
        self.subject_map = {} # index: subject name
        self.session_map = {} # index: session name
        self.label_map = {}   # {int(event_id): 'description'}
        self.event_id = {}    # {'event_name': int(event_id)}
        #
        self.ch_names = []
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
            old_labels = old_events[:, 2].copy()

            if sorted(list(old_event_id.values())) != list(range(len(old_event_id))):
                for old_event_name, old_event_label in old_event_id.items():
                    events[:, 2][old_labels == old_event_label] = \
                        fixed_event_id[old_event_name]
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
            self.label   = np.concatenate((self.label,   data.events[:, 2]))
            self.idx     = np.concatenate((self.idx,     range(epoch_len)))
            if len(self.data) == 0:
                self.data = data.get_data()
            else:
                self.data = np.concatenate((self.data,    data.get_data()))
            self.sfreq = data.info['sfreq']
            self.ch_names = data.info.ch_names.copy()

        self.session_map = {map_session[i]: i for i in map_session}
        self.subject_map = {map_subject[i]: i for i in map_subject}

    def copy(self) -> Epochs:
        """Return a copy of the object."""
        return deepcopy(self)

    # data splitting
    ## get list
    def get_subject_list(self) -> np.ndarray:
        """Return list of subject index of each epoch."""
        return self.subject

    def get_session_list(self) -> np.ndarray:
        """Return list of session index of each epoch."""
        return self.session

    def get_label_list(self) -> np.ndarray:
        """Return list of label index of each epoch."""
        return self.label

    ## get list by mask
    def get_subject_list_by_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return list of subject index of each epoch by mask.

        Args:
            mask: Mask to filter out remaining epochs. 1D np.ndarray of bool.
        """
        return self.subject[mask]

    def get_session_list_by_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return list of session index of each epoch by mask.

        Args:
            mask: Mask to filter out remaining epochs. 1D np.ndarray of bool.
        """
        return self.session[mask]

    def get_label_list_by_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return list of label index of each epoch by mask.

        Args:
            mask: Mask to filter out remaining epochs. 1D np.ndarray of bool.
        """
        return self.label[mask]

    def get_idx_list_by_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return list of epoch index of each epoch by mask.

        Args:
            mask: Mask to filter out remaining epochs. 1D np.ndarray of bool.
        """
        return self.idx[mask]

    ## get by index
    def get_subject_name(self, idx: int) -> str:
        """Return subject name by subject index.

        Args:
            idx: Subject index.
        """
        return self.subject_map[idx]

    def get_session_name(self, idx: int) -> str:
        """Return session name by session index.

        Args:
            idx: Session index.
        """
        return self.session_map[idx]

    def get_label_name(self, idx: int) -> str:
        """Return label name by label index.

        Args:
            idx: Label index.
        """
        return self.label_map[idx]

    ## get map
    def get_subject_map(self) -> dict:
        """Return mapping from subject index to subject name."""
        return self.subject_map

    def get_session_map(self) -> dict:
        """Return mapping from session index to session name."""
        return self.session_map

    def get_label_map(self) -> dict:
        """Return mapping from label index to label name."""
        return self.label_map

    ## misc getter
    def get_subject_index_list(self) -> list:
        """Return list of subject index."""
        return list(self.subject_map.keys())

    def pick_subject_mask_by_idx(self, idx: int) -> np.ndarray:
        """Return mask of epochs by subject index.

        Args:
            idx: Subject index.
        """
        return self.subject == idx

    ## data info
    def get_data_length(self) -> int:
        """Return number of total epochs."""
        return len(self.data)

    ## picker
    """
        How it works:
            (Enter pick_XXX)
            Get the list of selected attributes.
            (Enter _pick)
            Calculate the number of ids to be selected. (In _get_real_num)
            Generate the mask and selected counter filtered by each attribute.
                (In _generate_mask_target)
            while number of epochs to be selected > 0:
                Get the mask and counter of epochs with least selected counter.
                    (In _get_filtered_mask_pair)
                Choose one epoch
                Select all epochs matched the chosen epoch by the attribute.
                Update the counter of groups that contain the chosen epoch.
                    (In _update_mask_target)
                Decrease the number of ids to be selected.
            Return the selected mask.
        Note: sequence of attributes to be selected can make the result different.
            (The sequence is defined in _get_filtered_mask_pair)
        Note: Trial is different from other attributes.
            (In pick_trial) The index of trial is discarded
                            ecause it is meaningless so far.
    """
    def _generate_mask_target(self, mask: np.ndarray) -> dict:
        """Return mask-counter pair, group by label, subject, and session.

        Args:
            mask: Mask to filter out remaining epochs. 1D np.ndarray of bool.

        Returns:
            dict[label_idx][subject_idx][session_idx] = [target_filter_mask, count]
        """
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
                    filter_mask = (
                        (self.label == label_idx) &
                        (self.subject == subject_idx) &
                        (self.session == session_idx)
                    )
                    target_filter_mask = filter_mask & mask
                    filter_preview_mask[label_idx][subject_idx][session_idx] = \
                        [target_filter_mask, 0]
        return filter_preview_mask

    def _get_filtered_mask_pair(self, filter_preview_mask: dict) -> list:
        """Return mask-counter pair with least selected group.

        Args:
            filter_preview_mask:
                Mask-counter pair, group by label, subject, and session.

        Returns:
            [target_filter_mask, count] with least count.
        """

        min_count = self.get_data_length()
        filtered_mask_pair = None
        sequence = [
            TrialSelectionSequence.SESSION,
            TrialSelectionSequence.SUBJECT,
            TrialSelectionSequence.Label
        ]
        for a in np.unique(getattr(self, f"get_{sequence[0].value}_list")()):
            for b in np.unique(getattr(self, f"get_{sequence[1].value}_list")()):
                for c in np.unique(getattr(self, f"get_{sequence[2].value}_list")()):
                    args = [a, b, c]
                    label_idx = args[sequence.index(TrialSelectionSequence.Label)]
                    subject_idx = args[sequence.index(TrialSelectionSequence.SUBJECT)]
                    session_idx = args[sequence.index(TrialSelectionSequence.SESSION)]
                    target = filter_preview_mask[label_idx][subject_idx][session_idx]
                    if target[0].any() and target[1] < min_count:
                        min_count = target[1]
                        filtered_mask_pair = target
        return filtered_mask_pair

    def _update_mask_target(self, filter_preview_mask: dict, pos: np.ndarray) -> dict:
        """Update mask-counter pair by selected mask.

        Args:
            filter_preview_mask:
                Mask-counter pair, group by label, subject, and session.
            pos:
                Mask of selected epochs.

        Returns:
            Updated mask-counter pair.
        """
        for label_idx in filter_preview_mask:
            unique_subject_idx = filter_preview_mask[label_idx]
            for subject_idx in unique_subject_idx:
                unique_session_idx = unique_subject_idx[subject_idx]
                for session_idx in unique_session_idx:
                    filtered_mask_pair = unique_session_idx[session_idx]
                    filtered_mask_pair[1] += sum(filtered_mask_pair[0] & pos)
                    filtered_mask_pair[0] &= np.logical_not(pos)
        return filter_preview_mask

    def _get_real_num(self,
                     target_type: np.ndarray,
                     value: float | list[int],
                     split_unit: SplitUnit,
                     mask: np.ndarray,
                     clean_mask: np.ndarray,
                     group_idx: int) -> int:
        """Return number of epochs to be selected.

        Args:
            target_type:
                List of index of target type.
                Can be list index of subject or session.
            value: Value of splitting option.
                   Can be ratio, number, or list of manual selection.
            split_unit: SplitUnit of splitting option.
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs,
                        including all available selection.
                        1D np.ndarray of bool.
            group_idx: Group index of cross validation.
        """
        if clean_mask is None:
            target = len(np.unique(target_type[mask]))
        else:
            target = len(np.unique(target_type[clean_mask]))
        if split_unit == SplitUnit.KFOLD:
            inc = target % value
            num = target // value
            if inc > group_idx:
                num += 1
        elif split_unit == SplitUnit.RATIO:
            num = value * target
        elif split_unit == SplitUnit.NUMBER:
            num = min(value, target)
        else:
            raise NotImplementedError
        num = int(num)
        return num

    def _pick(self,
             target_type: np.ndarray,
             mask: np.ndarray,
             clean_mask: np.ndarray,
             value: float | list[int],
             split_unit: SplitUnit,
             group_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return mask of selected epochs by splitting option and target type.

        Args:
            target_type: List of index of target type.
                         Can be list index of subject or session.
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs,
                        including all available selection.
                        1D np.ndarray of bool.
            value: Value of splitting option.
                   Can be ratio, number, or list of manual selection.
            split_unit: SplitUnit of splitting option.
            group_idx: Group index of cross validation.

        Returns:
            [selected_mask, remaining_mask]
        """
        num = self._get_real_num(
            target_type, value, split_unit, mask, clean_mask, group_idx
        )
        ret = mask & False
        filter_preview_mask = self._generate_mask_target(mask)
        while num > 0:
            filtered_mask_pair = self._get_filtered_mask_pair(filter_preview_mask)
            if filtered_mask_pair is None:
                return ret, mask
            target = target_type[filtered_mask_pair[0]]
            if len(target) > 0:
                pos = (mask & (target_type == target[-1]))
                ret |= pos
                mask &= np.logical_not(pos)
                self._update_mask_target(filter_preview_mask, pos)
                filtered_mask_pair[0] &= np.logical_not(pos)
                filtered_mask_pair[1] += sum(pos)
                num -= 1
        return ret, mask

    def _pick_manual(self,
                    target_type: np.ndarray,
                    mask: np.ndarray,
                    value: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return mask of selected epochs by manual selection.

        Args:
            target_type: List of index of target type.
                         Can be list index of subject or session.
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            value: List of manual selection.

        Returns:
            [selected_mask, remaining_mask]
        """
        ret = mask & False
        for v in value:
            pos = (mask & (target_type == v))
            ret |= pos
            mask &= np.logical_not(pos)
        return ret, mask

    def pick_subject(self,
                     mask: np.ndarray,
                     clean_mask: np.ndarray,
                     value: float | list[int],
                     split_unit: SplitUnit,
                     group_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return mask of epochs selected by subject.

        Args:
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs,
                        including all available selection.
                        1D np.ndarray of bool.
            value: Value of splitting option.
                   Can be ratio, number, or list of manual selection.
            split_unit: SplitUnit of splitting option.
            group_idx: Group index of cross validation.

        Returns:
            [selected_mask, remaining_mask]
        """
        target_type = self.get_subject_list()
        if split_unit == SplitUnit.MANUAL:
            return self._pick_manual(target_type, mask, value)
        else:
            return self._pick(
                target_type, mask, clean_mask, value, split_unit, group_idx
            )

    def pick_session(self,
                     mask: np.ndarray,
                     clean_mask: np.ndarray,
                     value: float | list[int],
                     split_unit: SplitUnit,
                     group_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return mask of epochs selected by session.

        Args:
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs,
                        including all available selection.
                        1D np.ndarray of bool.
            value: Value of splitting option.
                   Can be ratio, number,
                   or list of manual selection.
            split_unit: SplitUnit of splitting option.
            group_idx: Group index of cross validation.

        Returns:
            [selected_mask, remaining_mask]
        """
        target_type = self.get_session_list()
        if split_unit == SplitUnit.MANUAL:
            return self._pick_manual(target_type, mask, value)
        else:
            return self._pick(
                target_type, mask, clean_mask, value, split_unit, group_idx
            )

    def pick_trial(self,
                   mask: np.ndarray,
                   clean_mask: np.ndarray,
                   value: float | list[int],
                   split_unit: SplitUnit,
                   group_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return mask of epochs selected by trial.

        Args:
            mask: Mask to filter out remaining epochs,
                  ecxluding already selected cross validation part.
                  1D np.ndarray of bool.
            clean_mask: Mask to filter out remaining epochs,
                        including all available selection.
                        1D np.ndarray of bool.
            value: Value of splitting option.
                   Can be ratio, number, or list of manual selection.
                   When split_unit is manual selection,
                   value is the boolean mask of selected epochs.
            split_unit: SplitUnit of splitting option.
            group_idx: Group index of cross validation.

        Returns:
            [selected_mask, remaining_mask]
        """
        ret = mask & False
        # manual selection
        if split_unit == SplitUnit.MANUAL:
            ret[value] = True
            ret &= mask
            mask &= np.logical_not(ret)
            return ret, mask

        # get number of epochs to be selected
        target = sum(mask) if clean_mask is None else sum(clean_mask)
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
        # select epochs
        filter_preview_mask = self._generate_mask_target(mask)
        while num > 0:
            filtered_mask_pair = self._get_filtered_mask_pair(filter_preview_mask)
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
        """Return args for model initialization."""
        return  {'n_classes': len(self.label_map),
                 'channels': len(self.ch_names),
                 'samples': self.data.shape[-1],
                 'sfreq': self.sfreq}

    def get_data(self) -> np.ndarray:
        """Return data."""
        return self.data

    #eval
    def get_label_number(self) -> int:
        """Return number of labels."""
        return len(self.label_map)

    def get_channel_names(self) -> list:
        """Return list of channel names."""
        return self.ch_names

    def get_epoch_duration(self) -> float:
        """Return duration of each epoch in seconds."""
        return np.round(self.data.shape[-1]/self.sfreq, 2)

    def set_channels(self, ch_names: list[str], channel_position: list) -> None:
        """Set channel names and positions.

        Args:
            ch_names: List of channel names.
            channel_position: List of channel positions. Position format is (x, y, z).
        """
        self.ch_names = ch_names
        self.channel_position = channel_position

    def get_montage_position(self) -> list:
        """Return list of channel positions."""
        return self.channel_position
