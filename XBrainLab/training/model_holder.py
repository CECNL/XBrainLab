import torch
ARG_DICT_SKIP_SET = set(['self', 'n_classes', 'channels', 'samples', 'sfreq'])

class ModelHolder:
    def __init__(self, target_model, model_params_map, pretrained_weight_path):
        self.target_model = target_model
        self.model_params_map = model_params_map
        self.pretrained_weight_path = pretrained_weight_path

    def get_model_desc_str(self):
        option_list = [f"{i}={self.model_params_map[i]}" for i in self.model_params_map if self.model_params_map[i] ]
        options = ', '.join(option_list)
        return f"{self.target_model.__name__} ({options})"

    def get_model(self, args):
        model = self.target_model(**self.model_params_map, **args)
        if self.pretrained_weight_path:
            model.load_state_dict(torch.load(self.pretrained_weight_path))
        return model
