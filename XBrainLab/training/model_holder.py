from __future__ import annotations

import torch


class ModelHolder:
    """Class for storing model information

    Holds the model class, model parameters, and pretrained weight path.

    Attributes:
        target_model (type): Model class, inherited from `torch.nn.Module`
        model_params_map (dict): Model parameters
        pretrained_weight_path (str): Path to pretrained weight
    """
    def __init__(
        self,
        target_model: type,
        model_params_map: dict,
        pretrained_weight_path: str | None = None
    ):
        self.target_model = target_model
        self.model_params_map = model_params_map
        self.pretrained_weight_path = pretrained_weight_path

    def get_model_desc_str(self) -> str:
        """Get model description string, including model name and parameters"""
        option_list = [
            f"{i}={self.model_params_map[i]}"
            for i in self.model_params_map if self.model_params_map[i]
        ]
        options = ', '.join(option_list)
        return f"{self.target_model.__name__} ({options})"

    def get_model(self, args) -> torch.nn.Module:
        """Get model instance with given parameters"""
        model = self.target_model(**self.model_params_map, **args)
        if self.pretrained_weight_path:
            model.load_state_dict(torch.load(self.pretrained_weight_path))
        return model
