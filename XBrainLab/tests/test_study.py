import mne
import pytest

from XBrainLab import Study
from XBrainLab.dataset import (
    Dataset,
    DatasetGenerator,
    DataSplittingConfig,
    TrainingType,
)
from XBrainLab.load_data import Raw, RawDataLoader
from XBrainLab.preprocessor import PreprocessBase
from XBrainLab.training import TRAINING_EVALUATION, ModelHolder, TrainingOption


def test_study_load_data():
    assert isinstance(Study().get_raw_data_loader(), RawDataLoader)

@pytest.fixture
def loaded_data_list():
    mne_data = mne.io.RawArray([[0]], mne.create_info(['test'], 100))
    return [Raw('test', mne_data)]

@pytest.fixture
def loaded_epoch_data_list():
    mne_data = mne.EpochsArray([[[0]]], mne.create_info(['test'], 100))
    return [Raw('test', mne_data)]

def _test_study_set_loaded_data_list_raise(study, loaded_data_list, force_update):
    if force_update:
        study.set_loaded_data_list(loaded_data_list, force_update)
    else:
        with pytest.raises(ValueError):
            study.set_loaded_data_list(loaded_data_list, force_update)
        study.clean_raw_data()
        study.set_loaded_data_list(loaded_data_list, force_update)

@pytest.mark.parametrize('force_update', [True, False])
def test_study_set_loaded_data_list(loaded_data_list, force_update):
    study = Study()
    study.set_loaded_data_list(loaded_data_list, force_update)
    _test_study_set_loaded_data_list_raise(study, loaded_data_list, force_update)

def _test_study_set_preprocessed_data_list_raise(study, loaded_data_list, force_update):
    if force_update or not study.datasets:
        study.set_preprocessed_data_list(loaded_data_list, force_update)
    else:
        with pytest.raises(ValueError):
            study.set_preprocessed_data_list(loaded_data_list, force_update)
        study.clean_datasets()
        study.set_preprocessed_data_list(loaded_data_list, force_update)


class FakePreprocessBase(PreprocessBase):
    def get_preprocess_desc(self):
        return 'test'
    def _data_preprocess(self, preprocessed_data):
        preprocessed_data.filepath = 'new'

@pytest.mark.parametrize('force_update', [True, False])
@pytest.mark.parametrize(
    'loaded_data_list_target, loaded_data_list_is_raw',
    [
        ("loaded_data_list", True),
        ("loaded_epoch_data_list", False)
    ]
)
@pytest.mark.parametrize(
    'test_hook',
    [_test_study_set_loaded_data_list_raise]
)
def test_study_set_preprocessed_data_list(
    loaded_data_list_target, loaded_data_list_is_raw, force_update,
    test_hook,
    request
):
    loaded_data_list = request.getfixturevalue(loaded_data_list_target)
    study = Study()
    study.set_loaded_data_list(loaded_data_list, force_update)
    _test_study_set_preprocessed_data_list_raise(study, loaded_data_list, force_update)
    if loaded_data_list_is_raw:
        assert study.epoch_data is None
    else:
        assert study.epoch_data is not None
    study.preprocess(FakePreprocessBase)
    assert study.preprocessed_data_list[0].get_filepath() == 'new'
    study.reset_preprocess()
    assert study.preprocessed_data_list[0].get_filepath() == 'test'
    test_hook(study, loaded_data_list, force_update)

def test_study_get_datasets_generator(loaded_epoch_data_list):
    config = DataSplittingConfig(
        TrainingType.FULL, is_cross_validation=False,
        val_splitter_list=[], test_splitter_list=[]
    )
    study = Study()
    study.set_loaded_data_list(loaded_epoch_data_list)
    assert isinstance(study.get_datasets_generator(config), DatasetGenerator)


def _test_study_set_datasets_raise(study, dataset, force_update):
    if force_update:
        study.set_datasets([dataset], force_update)
    else:
        with pytest.raises(ValueError):
            study.set_datasets([dataset], force_update)
        study.clean_datasets()
        study.set_datasets([dataset], force_update)

@pytest.mark.parametrize('force_update', [True, False])
@pytest.mark.parametrize(
    'loaded_data_list_target',
    ["loaded_data_list", "loaded_epoch_data_list"]
)
@pytest.mark.parametrize(
    'test_hook',
    [
        _test_study_set_loaded_data_list_raise,
        _test_study_set_preprocessed_data_list_raise
    ]
)
def test_study_set_datasets(
    loaded_data_list_target, force_update,
    loaded_epoch_data_list,
    test_hook,
    request
):
    loaded_data_list = request.getfixturevalue(loaded_data_list_target)
    config = DataSplittingConfig(
        TrainingType.FULL, is_cross_validation=False,
        val_splitter_list=[], test_splitter_list=[]
    )
    study = Study()
    study.set_loaded_data_list(loaded_epoch_data_list)
    dataset = Dataset(study.epoch_data, config)

    study.set_datasets([dataset], force_update)
    _test_study_set_datasets_raise(study, dataset, force_update)
    test_hook(study, loaded_data_list, force_update)


class FakeRecord:
    def export_csv(self, filepath):
        self.filepath = filepath

class FakePlan:
    def __init__(self, name, real_name):
        self.name = name
        self.real_name = real_name
    def get_eval_record(self): # pragma: no cover
        pass

class FakeTrainer:
    def __init__(self):
        self.running = False
        self.interact = None
        self.interrupt = False
        self.return_plan = False
    def run(self, interact=False):
        self.running = True
        self.interact = interact
    def set_interrupt(self):
        self.interrupt = True
    def is_running(self):
        return self.running
    def clean(self, force_update):
        pass
    def get_real_training_plan(self, name, real_name):
        if self.return_plan:
            return FakePlan(name, real_name)
        else:
            raise ValueError


@pytest.fixture
def trainer_study():
    study = Study()
    study.trainer = FakeTrainer()
    return study

@pytest.mark.parametrize('force_update', [True, False])
def test_study_set_training_option(trainer_study, force_update):
    option = TrainingOption(
        'test', int, 0, True, None, 1, 1, 1, 1, TRAINING_EVALUATION.TEST_ACC, 1
    )
    if force_update:
        trainer_study.set_training_option(option, force_update)
    else:
        with pytest.raises(ValueError):
            trainer_study.set_training_option(option, force_update)
        trainer_study.clean_trainer()
        trainer_study.set_training_option(option, force_update)

@pytest.mark.parametrize('force_update', [True, False])
def test_study_set_model_holder(trainer_study, force_update):
    holder = ModelHolder(int, 0)
    if force_update:
        trainer_study.set_model_holder(holder, force_update)
    else:
        with pytest.raises(ValueError):
            trainer_study.set_model_holder(holder, force_update)
        trainer_study.clean_trainer()
        trainer_study.set_model_holder(holder, force_update)

@pytest.mark.parametrize('force_update', [True, False])
def test_study_generate_plan(mocker, trainer_study, force_update):
    holder_mock = mocker.patch(
        'XBrainLab.training.TrainingPlanHolder.__init__', return_value=None
    )
    trainer_mock = mocker.patch(
        'XBrainLab.training.Trainer.__init__', return_value=None
    )
    trainer_study.datasets = [1, 2, 3]
    trainer_study.training_option = 2
    trainer_study.model_holder = 3
    if force_update:
        trainer_study.generate_plan(force_update=force_update)
    else:
        with pytest.raises(ValueError):
            trainer_study.generate_plan(force_update=force_update)
        trainer_study.clean_trainer()
        trainer_study.generate_plan(force_update=force_update)

    called_args_list = holder_mock.call_args_list
    assert len(called_args_list) == 3
    for i in range(3):
        called_args = called_args_list[i][0]
        assert called_args[0] == 3
        assert called_args[1] == (i + 1)
        assert called_args[2] == 2

    trainer_mock.assert_called_once()

@pytest.mark.parametrize(
    'missing_part, complain',
    [
        ['datasets', 'dataset'],
        ['training_option', 'training option'],
        ['model_holder', 'model holder']
    ]
)
def test_study_generate_plan_missing_options(missing_part, complain):
    study = Study()
    study.datasets = [1, 2, 3]
    study.training_option = 2
    study.model_holder = 3
    setattr(study, missing_part, None)

    with pytest.raises(ValueError, match=f".*{complain}.*"):
        study.generate_plan()

def test_study_training(trainer_study):
    assert not trainer_study.is_training()
    trainer_study.train()
    assert trainer_study.is_training()
    trainer_study.stop_training()
    assert trainer_study.trainer.interrupt

def test_study_training_not_set():
    study = Study()
    assert not study.is_training()
    with pytest.raises(ValueError):
        study.train()
    assert not study.is_training()
    with pytest.raises(ValueError):
        study.stop_training()

@pytest.mark.parametrize('has_record', [True, False])
@pytest.mark.parametrize('has_eval', [True, False])
def test_study_export_output_csv(mocker, trainer_study, has_record, has_eval):
    record = FakeRecord()
    return_value = None
    if has_eval:
        return_value = record
    mocker.patch(
        'XBrainLab.tests.test_study.FakePlan.get_eval_record',
        return_value=return_value
    )
    trainer_study.trainer.return_plan = has_record
    if not has_record:
        with pytest.raises(ValueError):
            trainer_study.export_output_csv('test', '1', '2')
        return
    if not has_eval:
        with pytest.raises(ValueError):
            trainer_study.export_output_csv('test', '1', '2')
        return
    trainer_study.export_output_csv('test', '1', '2')
    assert record.filepath == 'test'


def test_study_export_output_csv_not_set():
    study = Study()
    with pytest.raises(ValueError):
        study.export_output_csv('test', 'test', 'test')

def test_study_set_channels():
    class FakeEpochData:
        def set_channels(self, channels, channel_types):
            self.channels = channels
            self.channel_types = channel_types
    study = Study()
    study.epoch_data = FakeEpochData()
    study.set_channels([1], [2])
    assert study.epoch_data.channels == [1]
    assert study.epoch_data.channel_types == [2]

def test_study_set_channels_not_set():
    study = Study()
    with pytest.raises(ValueError):
        study.set_channels([], [])
