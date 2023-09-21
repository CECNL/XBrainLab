from XBrainLab.training import ModelHolder


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state_dict = None

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

def test_model_holder(mocker):
    target_model = FakeModel
    model_params_map = {
        'a': 1,
        'b': 2
    }
    pretrained_weight_path = 'test.pth'
    holder = ModelHolder(target_model, model_params_map, pretrained_weight_path)
    mocker.patch('torch.load', return_value='state_dict')
    model = holder.get_model({'c': 3})

    assert holder.get_model_desc_str() == 'FakeModel (a=1, b=2)'
    assert model.kwargs == {'a': 1, 'b': 2, 'c': 3}
    assert model.state_dict == 'state_dict'
