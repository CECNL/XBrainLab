from XBrainLab.training.training_plan import to_holder, TrainRecord, ModelHolder, TrainingPlanHolder, TrainingOption
from XBrainLab.training.record import RecordKey
from XBrainLab.training.option import TRAINING_EVALUATION
from XBrainLab.dataset import DataSplitter, SplitByType, ValSplitByType, SplitUnit, DatasetGenerator
from XBrainLab.dataset import Epochs, SplitUnit, DataSplittingConfig, TrainingType
from XBrainLab.load_data import Raw
from XBrainLab.utils import set_seed

import mne
import torch
import pytest
import numpy as np
import time
CLASS_NUM = 4
ERROR_NUM = 3
SAMPLE_NUM = CLASS_NUM
REPEAT = 5
TOTAL_NUM = SAMPLE_NUM * REPEAT
BS = 2

class FakeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(CLASS_NUM, CLASS_NUM)
        self.my_state_dict = None

    def load_state_dict(self, state_dict):
        self.my_state_dict = state_dict

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    

@pytest.fixture
def y():
    return np.arange(SAMPLE_NUM).repeat(REPEAT)

def _create_raw(y, subject, session):
    """
    X = [[[1, 0, 0, 0]],
         [[1, 0, 0, 0]],
         [[1, 0, 0, 0]],
          ...
         [[0, 0, 0, 1]]]
    y = [0, 0, 0, 0, 0, 1, ...]
    """
    events = np.zeros((TOTAL_NUM, 3), dtype=int)
    events[:, 0] = np.arange(CLASS_NUM * REPEAT)
    events[:, 2] = y.copy()

    ch_types = 'eeg'
    ch_names = ['C1']
    event_id = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3}
    fs = 1
    info = mne.create_info(ch_names=ch_names,
                    sfreq=fs,
                    ch_types=ch_types)
    data = np.zeros((len(events), len(ch_names), CLASS_NUM))
    for idx, gt in enumerate(y):
        data[idx, 0, gt] = gt
            
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, event_id=event_id)
    raw = Raw(f"test/sub-{subject}_ses-{session}.fif", epochs)
    raw.set_subject_name(subject)
    raw.set_session_name(session)
    return raw

@pytest.fixture
def preprocessed_data_list(y):
    return [_create_raw(y, '01', '01'), 
            _create_raw(y, '02', '01'), 
            _create_raw(y, '03', '01')
    ]

@pytest.fixture
def epochs(preprocessed_data_list):
    return Epochs(preprocessed_data_list)

@pytest.fixture
def dataset(epochs):
    test_split_list = [
        DataSplitter(SplitByType.SUBJECT, '1', SplitUnit.NUMBER, True)
    ]
    val_split_list = [
        DataSplitter(ValSplitByType.SUBJECT, '1', SplitUnit.NUMBER, True)
    ]
    config = DataSplittingConfig(TrainingType.FULL, False, val_split_list, test_split_list)
    generator = DatasetGenerator(epochs, config)
    dataset = generator.generate()[0]
    return dataset

@pytest.fixture
def model_holder():
    args = {}
    path = None
    return ModelHolder(FakeModel, args, path)

@pytest.fixture
def training_option():
    args = {
        'output_dir': 'ok',
        'optim': torch.optim.Adam,
        'optim_params': {},
        'use_cpu': True,
        'gpu_idx': None,
        'epoch': 10,
        'bs': BS,
        'lr': 0.01,
        'checkpoint_epoch': 2, 
        'evaluation_option': TRAINING_EVALUATION.VAL_LOSS,
        'repeat_num': 5
    }
    return TrainingOption(**args)

@pytest.fixture
def export_mocker(mocker):
    mocker.patch('torch.save')
    mocker.patch('os.makedirs')

@pytest.fixture
def base_holder(export_mocker, model_holder, dataset, training_option):
    args = {
        'model_holder': model_holder,
        'dataset': dataset,
        'option': training_option
    }
    return TrainingPlanHolder(**args)



@pytest.mark.parametrize("test_arg", [
    'model_holder', 'dataset', 'option', None
])
def test_training_plan_holder_check_data(export_mocker, model_holder, dataset, training_option, test_arg):
    args = {
        'model_holder': model_holder,
        'dataset': dataset,
        'option': training_option
    }
    if test_arg is None:
        holder = TrainingPlanHolder(**args)
        assert len(holder.train_record_list) == REPEAT
        for record in holder.train_record_list:
            assert isinstance(record, TrainRecord)
    else:
        args[test_arg] = None
        with pytest.raises(ValueError):
            TrainingPlanHolder(**args)

def test_training_plan_holder_get_loader(base_holder):
    set_seed(0)
    trainHolder, valHolder, testHolder = base_holder.get_loader()
    assert isinstance(trainHolder, torch.utils.data.DataLoader)
    assert isinstance(valHolder, torch.utils.data.DataLoader)
    assert isinstance(testHolder, torch.utils.data.DataLoader)
    
    train_data = next(iter(trainHolder))
    assert train_data[0].shape == (BS, 1, CLASS_NUM)
    assert train_data[1].shape == (BS,)
    val_data = next(iter(valHolder))
    assert val_data[0].shape == (BS, 1, CLASS_NUM)
    assert val_data[1].shape == (BS,)
    test_data = next(iter(testHolder))
    assert test_data[0].shape == (BS, 1, CLASS_NUM)
    assert test_data[1].shape == (BS,)
    
    torch.testing.assert_close(test_data[0], val_data[0])
    torch.testing.assert_close(test_data[1], val_data[1])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(test_data[0], train_data[0])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(test_data[1], train_data[1])

@pytest.mark.parametrize("val_loader, test_loader, expected_loader", [
    ('val', 'test', 'test'),
    (None, 'test', 'test'),
    ('val', None, 'val'),
    (None, None, None),
])
def test_training_plan_holder_get_eval_loader(base_holder, dataset, model_holder, training_option, 
                                     val_loader, test_loader, expected_loader):
    repeat = 0
    seed = set_seed()
    model = model_holder.get_model({})
    training_option.evaluation_option = TRAINING_EVALUATION.VAL_LOSS
    record = TrainRecord(repeat=repeat, dataset=dataset, model=model, option=training_option, seed=seed)
    
    _, target_loader = base_holder.get_eval_pair(record, val_loader, test_loader)
    assert target_loader == expected_loader

@pytest.mark.parametrize("evaluation_option, state_dict_attr_name", [
    (TRAINING_EVALUATION.VAL_LOSS, f'best_val_{RecordKey.LOSS}_model'),
    (TRAINING_EVALUATION.VAL_LOSS, f'best_val_{RecordKey.LOSS}_model'),
    (TRAINING_EVALUATION.TEST_AUC, f'best_test_{RecordKey.AUC}_model'),
    (TRAINING_EVALUATION.TEST_AUC, f'best_test_{RecordKey.AUC}_model'),
    (TRAINING_EVALUATION.TEST_ACC, f'best_test_{RecordKey.ACC}_model'),
    (TRAINING_EVALUATION.TEST_ACC, f'best_test_{RecordKey.ACC}_model'),
])
@pytest.mark.parametrize("expected", ['test', None])
def test_training_plan_holder_get_eval_model(base_holder, dataset, model_holder, training_option,
                                    evaluation_option, state_dict_attr_name, expected):
    repeat = 0
    val_loader = None
    test_loader = None
    seed = set_seed()
    model = model_holder.get_model({})
    training_option.evaluation_option = evaluation_option
    record = TrainRecord(repeat=repeat, dataset=dataset, model=model, option=training_option, seed=seed)
    if expected:
        expected = np.random.rand(1)
    setattr(record, state_dict_attr_name, expected)
    
    target_model, _ = base_holder.get_eval_pair(record, val_loader, test_loader)
    if expected:
        assert isinstance(target_model, FakeModel)
        assert target_model.my_state_dict == expected
    else:
        assert target_model is None

@pytest.mark.parametrize("val_loader, test_loader, expected_loader", [
    ('val', 'test', 'test'),
    (None, 'test', 'test'),
    ('val', None, 'val'),
    (None, None, None),
])
@pytest.mark.parametrize("evaluation_option", [
    i for i in TRAINING_EVALUATION
] + [None])
def test_training_plan_holder_get_eval_pair_not_implemented(base_holder, dataset, model_holder, training_option, 
                                     val_loader, test_loader, expected_loader, evaluation_option):
    repeat = 0
    seed = set_seed()
    model = model_holder.get_model({})
    training_option.evaluation_option = evaluation_option
    record = TrainRecord(repeat=repeat, dataset=dataset, model=model, option=training_option, seed=seed)
    
    if evaluation_option:
        _, target_loader = base_holder.get_eval_pair(record, val_loader, test_loader)
        assert target_loader == expected_loader
    else:
        with pytest.raises(NotImplementedError):
            base_holder.get_eval_pair(record, val_loader, test_loader)

def test_training_plan_holder_get_eval_model_by_lastest_model(mocker, base_holder, dataset, model_holder, training_option):
    repeat = 0
    val_loader = None
    test_loader = None
    seed = set_seed()
    model = model_holder.get_model({})
    mocker.patch.object(model, 'state_dict', return_value='test')
    training_option.evaluation_option = TRAINING_EVALUATION.LAST_EPOCH
    record = TrainRecord(repeat=repeat, dataset=dataset, model=model, option=training_option, seed=seed)

    target_model, _ = base_holder.get_eval_pair(record, val_loader, test_loader)

    assert isinstance(target_model, FakeModel)
    assert target_model.my_state_dict == 'test'

def test_training_plan_holder_set_interrupt(base_holder):
    assert base_holder.interrupt == False
    base_holder.set_interrupt()
    assert base_holder.interrupt == True
    base_holder.clear_interrupt()
    assert base_holder.interrupt == False

def test_training_plan_holder_trivial_getter(base_holder, dataset):
    assert base_holder.get_name() == "0-Group_0"
    assert base_holder.get_dataset() == dataset
    assert len(base_holder.get_plans()) == REPEAT

@pytest.mark.timeout(10)
@pytest.mark.parametrize("interrupt", [True, False])
def test_training_plan_holder_one_epoch(mocker, base_holder, interrupt):
    model = base_holder.model_holder.get_model({})
    trainLoader, valLoader, testLoader = base_holder.get_loader()
    train_record = base_holder.train_record_list[0]
    optimizer = train_record.optim
    criterion = train_record.criterion
    
    update_train_mock = mocker.patch.object(train_record, 'update_train')
    update_val_mock = mocker.patch.object(train_record, 'update_eval')
    update_test_mock = mocker.patch.object(train_record, 'update_test')
    update_statistic_mock = mocker.patch.object(train_record, 'update_statistic')
    export_checkpoint_mock = mocker.patch.object(train_record, 'export_checkpoint')
    fake_test_result = {'test': 'test'}
    mocker.patch('XBrainLab.training.training_plan._test_model', return_value=fake_test_result)
    if interrupt:
        base_holder.set_interrupt()

    start_time = time.time()
    base_holder.train_one_epoch(model, trainLoader, valLoader, testLoader, optimizer, criterion, train_record)
    total_time = time.time() - start_time

    if interrupt:
        assert update_train_mock.call_count == 0
        assert update_val_mock.call_count == 0
        assert update_test_mock.call_count == 0
        assert update_statistic_mock.call_count == 0
        assert export_checkpoint_mock.call_count == 0
        return

    update_train_mock.assert_called_once()
    update_val_mock.assert_called_once()
    update_test_mock.assert_called_once()
    update_statistic_mock.assert_called_once()
    export_checkpoint_mock.assert_not_called()
    
    step_called_args = update_statistic_mock.call_args[0]
    assert (step_called_args[0]["time"]  - total_time) < 0.1
    assert step_called_args[0]["lr"] == 0.01

    update_val_called_args = update_val_mock.call_args[0][0]
    assert update_val_called_args == fake_test_result
    
    update_test_called_args = update_test_mock.call_args[0][0]
    assert update_test_called_args == fake_test_result

    base_holder.train_one_epoch(model, trainLoader, valLoader, testLoader, optimizer, criterion, train_record)
    export_checkpoint_mock.assert_called_once()

@pytest.mark.timeout(10)
def test_training_plan_holder_train_one_repeat(mocker, base_holder):
    train_record = base_holder.train_record_list[0]
    
    def set_interrupt(*args, **kwargs):
        base_holder.set_interrupt()
    train_one_epoch_mock = mocker.patch.object(base_holder, 'train_one_epoch')
    train_one_epoch_mock.side_effect = set_interrupt
    export_checkpoint_mock = mocker.patch.object(train_record, 'export_checkpoint')

    base_holder.train_one_repeat(train_record)

    train_one_epoch_mock.assert_called_once()
    export_checkpoint_mock.assert_called_once()

# check status
@pytest.mark.timeout(10)
def test_training_plan_holder_train_one_repeat_status(mocker, base_holder):
    original_train_one_epoch = base_holder.train_one_epoch
    epoch_counter = 0
    def train_one_epoch_side_effect(*args, **kwargs):
        nonlocal epoch_counter
        assert base_holder.get_training_status().startswith("Training")
        assert base_holder.get_training_epoch() == epoch_counter
        assert base_holder.get_epoch_progress_text() == str(epoch_counter) + " / 50"
        assert base_holder.is_finished() == False
        original_train_one_epoch(*args, **kwargs)
        epoch_counter += 1
        assert base_holder.get_training_epoch() == epoch_counter
        assert base_holder.get_epoch_progress_text() == str(epoch_counter) + " / 50"
        for i in base_holder.get_training_evaluation():
            assert i != "-"
    train_one_epoch_mock = mocker.patch.object(base_holder, 'train_one_epoch')
    train_one_epoch_mock.side_effect = train_one_epoch_side_effect

    train_record = base_holder.train_record_list[0]
    for i in base_holder.get_training_evaluation():
        assert i == "-"
    base_holder.train_one_repeat(train_record)
   

@pytest.mark.timeout(10)
def test_training_plan_holder_train_one_repeat_empty_training_data(mocker, base_holder):
    train_record = base_holder.train_record_list[0]
    get_loader_mock = mocker.patch.object(base_holder, 'get_loader', return_value=(None, None, None))
    with pytest.raises(ValueError):
        base_holder.train_one_repeat(train_record)


@pytest.mark.timeout(10)
def test_training_plan_holder_train_one_repeat_eval(mocker, base_holder):
    train_record = base_holder.train_record_list[0]

    set_eval_record_mock = mocker.patch.object(train_record, 'set_eval_record')
    base_holder.train_one_repeat(train_record)

    set_eval_record_mock.assert_called_once()

@pytest.mark.timeout(10)
def test_training_plan_holder_train_one_repeat_already_finished(mocker, base_holder):
    train_record = base_holder.train_record_list[0]
    
    mocker.patch.object(train_record, 'is_finished', return_value=True)
    train_one_epoch_mock = mocker.patch.object(base_holder, 'train_one_epoch')
    export_checkpoint_mock = mocker.patch.object(train_record, 'export_checkpoint')
    set_eval_record_mock = mocker.patch.object(train_record, 'set_eval_record')

    base_holder.train_one_repeat(train_record)
    assert train_one_epoch_mock.call_count == 0
    assert export_checkpoint_mock.call_count == 0
    assert set_eval_record_mock.call_count == 0

@pytest.mark.timeout(10)
def test_training_plan_holder_train(mocker, base_holder):
    original_train_one_repeat = base_holder.train_one_repeat
    
    train_one_repeat_mock = mocker.patch.object(base_holder, 'train_one_repeat')
    repeat_counter = 0
    def train_one_repeat_side_effect(*args, **kwargs):
        nonlocal repeat_counter
        assert base_holder.get_training_status().startswith("Initializing")
        assert base_holder.get_training_repeat() == repeat_counter
        assert base_holder.is_finished() == False
        original_train_one_repeat(*args, **kwargs)
        repeat_counter += 1
    train_one_repeat_mock.side_effect = train_one_repeat_side_effect

    original_get_eval_record = base_holder.get_eval_pair
    get_eval_pair_mock = mocker.patch.object(base_holder, 'get_eval_pair')
    def get_eval_pair_side_effect(*args, **kwargs):
        assert base_holder.get_training_status().startswith("Evaluating")
        return original_get_eval_record(*args, **kwargs)
    get_eval_pair_mock.side_effect = get_eval_pair_side_effect

    assert base_holder.get_training_status() == "Pending"
    assert base_holder.is_finished() == False
    assert base_holder.get_training_repeat() == 0
    assert base_holder.get_training_epoch() == 0
    for i in base_holder.get_training_evaluation():
        assert i == "-"
    assert base_holder.get_epoch_progress_text() == "0 / 50"
    base_holder.train()
    assert base_holder.get_training_status() == "Finished"
    assert base_holder.is_finished() == True
    assert base_holder.get_training_repeat() == 4
    assert base_holder.get_training_epoch() == 10
    for i in base_holder.get_training_evaluation():
        assert i != "-"
    assert base_holder.get_epoch_progress_text() == "50 / 50"
    train_one_repeat_mock.assert_called()
    get_eval_pair_mock.assert_called()

@pytest.mark.timeout(10)
def test_training_plan_holder_train_status(mocker, base_holder):
    original_train_one_repeat = base_holder.train_one_repeat
    train_one_repeat_mock = mocker.patch.object(base_holder, 'train_one_repeat')
    def train_one_repeat_side_effect(*args, **kwargs):
        base_holder.set_interrupt()
        original_train_one_repeat(*args, **kwargs)
    train_one_repeat_mock.side_effect = train_one_repeat_side_effect

    base_holder.train()
    assert base_holder.is_finished() == False
    assert base_holder.get_training_status() == "Pending"
    train_one_repeat_mock.assert_called()

@pytest.mark.timeout(10)
def test_training_plan_holder_train_error(mocker, base_holder):
    train_one_repeat_mock = mocker.patch.object(base_holder, 'train_one_repeat')
    def train_one_repeat_side_effect(*args, **kwargs):
        raise RuntimeError("test")
    train_one_repeat_mock.side_effect = train_one_repeat_side_effect

    base_holder.train()
    assert base_holder.is_finished() == False
    assert base_holder.get_training_status() == "test"
    train_one_repeat_mock.assert_called()
    