from typing import List

from ..load_data import Raw
from .base import PreprocessBase


class ChannelSelection(PreprocessBase):
    """Preprocessing class for selecting channels.

    Input:
        selected_channels: List of names of selected channels.
    """

    def get_preprocess_desc(self, selected_channels: List[str]):
        return f"Select {len(selected_channels)} Channel"

    def _data_preprocess(self, preprocessed_data: Raw, selected_channels: List[str]):
        # Check if channel is selected
        if len(selected_channels) == 0:
            raise ValueError("No Channel is Selected")
        preprocessed_data.get_mne().pick_channels(selected_channels)
