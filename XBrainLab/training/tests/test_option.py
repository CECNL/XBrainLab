from XBrainLab.training import parse_device_name, parse_optim_name, TRAINING_EVALUATION, TrainingOption, TestOnlyOption

import pytest
import torch

@pytest.mark.parametrize("use_cpu, gpu_idx, expected",[
    (True, None, "cpu"),
    (True, 0, "cpu"),
    (False, 0, "0 - test"),
    (False, 1, "1 - test"),
    (False, None, None),
])
def test_parse_device_name(mocker, use_cpu, gpu_idx, expected):
    mocker.patch("torch.cuda.get_device_name", return_value="test")
    if expected is None:
        with pytest.raises(ValueError):
            parse_device_name(use_cpu, gpu_idx)
    else:
        assert parse_device_name(use_cpu, gpu_idx) == expected

class FakeOptim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

def test_parse_optim_name():
    target = FakeOptim
    params_map = {
        'a': 1,
        'b': 2
    }
    assert parse_optim_name(target, params_map) == 'FakeOptim (a=1, b=2)'

@pytest.mark.parametrize("kwargs, has_error",[
    ({'output_dir': None}, True),
    ({'optim': None}, True),
    ({'optim_params': None}, True),
    
    ({'use_cpu': None, 'gpu_idx': None}, True),
    ({'use_cpu': None, 'gpu_idx': 1}, True),
    ({'use_cpu': False, 'gpu_idx': None}, True),
    ({'use_cpu': False, 'gpu_idx': 1}, False),
    ({'use_cpu': False, 'gpu_idx': 'cuda:0'}, True),
    ({'use_cpu': True, 'gpu_idx': None}, False),
    ({'use_cpu': True, 'gpu_idx': 1}, False),

    ({'epoch': 10.5}, False),
    ({'epoch': 10}, False),
    ({'epoch': -5}, False),
    ({'epoch': "error"}, True),
    ({'epoch': None}, True),
    ({'bs': None}, True),
    ({'bs': "error"}, True),
    ({'lr': None}, True),
    ({'lr': "error"}, True),
    ({'checkpoint_epoch': None}, True),
    ({'checkpoint_epoch': 0}, False),
    ({'checkpoint_epoch': "error"}, True),
    ({'evaluation_option': None}, True),
    ({'repeat_num': None}, True),
    ({'repeat_num': "error"}, True),
])
def test_option(kwargs, has_error):
    args = {
        'output_dir': 'ok',
        'optim': FakeOptim,
        'optim_params': {'a': 1, 'b': 2}, 
        'use_cpu': False,
        'gpu_idx': 0,
        'epoch': 10,
        'bs': 20, 
        'lr': 0.01,
        'checkpoint_epoch': 10, 
        'evaluation_option': TRAINING_EVALUATION.VAL_LOSS,
        'repeat_num': 5
    }
    
    for k in kwargs:
        args[k] = kwargs[k]

    if has_error:
        with pytest.raises(ValueError):
            option = TrainingOption(**args)
        return
    
    option = TrainingOption(**args)


    assert option.get_output_dir() == 'ok'
    assert option.get_evaluation_option_repr() == "TRAINING_EVALUATION.VAL_LOSS"
    if args['use_cpu'] or (not args['use_cpu'] and torch.cuda.is_available()):
        assert option.get_device_name() == parse_device_name(args['use_cpu'], args['gpu_idx'])
    assert option.get_device() == "cpu" if args['use_cpu'] else "cuda:" + str(args['gpu_idx'])

    assert option.get_optim_name() == "FakeOptim"
    assert option.get_optim_desc_str() == parse_optim_name(FakeOptim, args["optim_params"])

    model = FakeModel()
    optim_instance = option.get_optim(model)
    assert isinstance(optim_instance, FakeOptim)
    
    for k in args['optim_params']:
        assert k in optim_instance.kwargs and optim_instance.kwargs[k] == args['optim_params'][k]
    assert optim_instance.kwargs['lr'] == args['lr']
    
    model_params = optim_instance.kwargs['params']
    expected_model_params = model.parameters()
    for p, e in zip(model_params, expected_model_params):
        torch.testing.assert_close(p, e)


@pytest.mark.parametrize("kwargs, has_error",[
    ({'output_dir': None}, True),
    
    ({'use_cpu': None, 'gpu_idx': None}, True),
    ({'use_cpu': None, 'gpu_idx': 1}, True),
    ({'use_cpu': False, 'gpu_idx': None}, True),
    ({'use_cpu': False, 'gpu_idx': 1}, False),
    ({'use_cpu': False, 'gpu_idx': 'cuda:0'}, True),
    ({'use_cpu': True, 'gpu_idx': None}, False),
    ({'use_cpu': True, 'gpu_idx': 1}, False),

    ({'bs': None}, True),
    ({'bs': "error"}, True),
])
def test_test_only_option(kwargs, has_error):
    args = {
        'output_dir': 'ok',
        'use_cpu': False,
        'gpu_idx': 0,
        'bs': 20
    }
    
    for k in kwargs:
        args[k] = kwargs[k]

    if has_error:
        with pytest.raises(ValueError):
            option = TestOnlyOption(**args)
        return
    
    option = TestOnlyOption(**args)


    assert option.get_output_dir() == 'ok'
    assert option.get_evaluation_option_repr() == "TRAINING_EVALUATION.LAST_EPOCH"
    
    if args['use_cpu'] or (not args['use_cpu'] and torch.cuda.is_available()):
       assert option.get_device_name() == parse_device_name(args['use_cpu'], args['gpu_idx'])
    assert option.get_device() == "cpu" if args['use_cpu'] else "cuda:" + str(args['gpu_idx'])

    assert option.get_optim_name() == "-"
    assert option.get_optim_desc_str() == "-"

    assert option.get_optim(None) == None
    assert option.get_optim(10) == None
