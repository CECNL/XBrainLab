from .base import PreprocessBase

class ChannelSelection(PreprocessBase):

    def get_preprocess_desc(self, selected_channels):
        return f"Select {len(selected_channels)} Channel"

    def _data_preprocess(self, preprocessed_data, selected_channels):
        # Check if channel is selected
        if len(selected_channels) == 0:
            raise ValueError("No Channel is Selected")

        preprocessed_data.get_mne().pick_channels(selected_channels)
