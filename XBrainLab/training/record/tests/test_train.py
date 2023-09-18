from XBrainLab.training.record import TrainRecord, TrainRecordKey, RecordKey, EvalRecord
from XBrainLab.utils import set_seed

from ...tests.test_training_plan import (
    export_mocker, model_holder, dataset, training_option, # noqa: F401
    epochs, preprocessed_data_list, y, # noqa: F401
    FakeModel, CLASS_NUM
)

import pytest
import numpy as np
import os
import torch
from matplotlib import pyplot as plt

def test_train_record(mocker, dataset, training_option, model_holder): # noqa: F811
    repeat = 0
    seed = set_seed(0)
    model = model_holder.get_model({})
    create_dir_mock = mocker.patch.object(TrainRecord, 'create_dir')
    TrainRecord(repeat, dataset, model, training_option, seed)
    create_dir_mock.assert_called_once()

def test_train_record_getter(
    export_mocker, dataset, training_option, model_holder # noqa: F811
):
    repeat = 0
    seed = set_seed(0)
    model = model_holder.get_model({})
    record = TrainRecord(repeat, dataset, model, training_option, seed)
    assert record.get_name() == "Repeat-0"
    assert record.get_epoch() == 0
    
    with pytest.raises(Exception):
        record.get_training_model('error')
    assert isinstance(record.get_training_model('cpu'), FakeModel)
    assert record.is_finished() is False

    for i in range(training_option.epoch):
        record.step()
        assert record.get_epoch() == i + 1
        assert record.is_finished() is False
    assert record.is_finished() is False
    record.set_eval_record("test")
    assert record.is_finished()
      
@pytest.fixture()
def cleanup():
    yield
    import shutil
    if os.path.exists('ok'):
        shutil.rmtree('ok')

def test_train_record_create_dir(
    cleanup, mocker, dataset, training_option, model_holder # noqa: F811
):
    repeat = 0
    seed = set_seed(0)
    model = model_holder.get_model({})
    record = TrainRecord(repeat, dataset, model, training_option, seed)
    expected = os.path.join(
        training_option.get_output_dir(), dataset.get_name(), record.get_name()
    )
    assert os.path.exists(expected)

def test_train_record_backup_dir(
    cleanup, mocker, dataset, training_option, model_holder # noqa: F811
):
    mkdir = os.makedirs
    move_mock = mocker.patch('shutil.move')
    make_dir_mock = mocker.patch('os.makedirs')
    repeat = 0
    seed = set_seed(0)
    model = model_holder.get_model({})
    record = TrainRecord(repeat, dataset, model, training_option, seed)
    
    record_name = record.dataset.get_name()
    repeat_name = record.get_name()
    output_root = record.option.get_output_dir()
    expected = os.path.join(output_root, record_name, repeat_name)
    expected_backup_root = os.path.join(output_root, record_name, 'backup')
    
    make_dir_mock.assert_called_once_with(expected)

    os.makedirs = mkdir
    os.makedirs(expected)
    make_dir_mock = mocker.patch('os.makedirs')
    record.create_dir()
    assert make_dir_mock.call_count == 2
    move_mock.assert_called_once()

    called_args_list = make_dir_mock.call_args_list
    bakup_create = called_args_list[0]
    assert bakup_create[0][0] == expected_backup_root
    new_create = called_args_list[1]
    assert new_create[0][0] == expected

    called_move_mock_args = move_mock.call_args[0]
    assert called_move_mock_args[0] == expected
    assert called_move_mock_args[1].startswith(
        os.path.join(expected_backup_root, repeat_name)
    )

@pytest.fixture()
def train_record(export_mocker, dataset, training_option, model_holder): # noqa: F811
    repeat = 0
    seed = set_seed(0)
    model = model_holder.get_model({})
    record = TrainRecord(repeat, dataset, model, training_option, seed)
    return record

def test_train_record_append_record(train_record):
    arr = []
    values = np.random.rand(10)
    for value in values:
        train_record.append_record(value, arr)
        assert len(arr) == 1
        assert arr[0] == value

def test_train_record_append_record_with_step(train_record):
    arr = []
    values = np.random.rand(10)
    for idx, value in enumerate(values):
        train_record.append_record(value, arr)
        train_record.step()
        assert len(arr) == idx + 1
        assert arr[-1] == value

def test_train_record_append_record_with_large_array(train_record):
    arr = [1, 2, 3, 4, 5]
    values = np.random.rand(10)
    for idx, value in enumerate(values):
        train_record.append_record(value, arr)
        train_record.step()
        if idx < 5:
            assert len(arr) == 5
        else:
            assert len(arr) == idx + 1
        assert arr[idx] == value

def test_train_record_append_record_with_small_array(train_record):
    arr = []
    for _ in range(5):
        train_record.step()
    train_record.append_record(15, arr)
    assert len(arr) == 6
    assert arr[-1] == 15
    for i in range(5):
        assert arr[i] is None

@pytest.mark.parametrize("update_type, test_result_key, expected_key", [
    ('val', TrainRecordKey.LOSS, 'best_val_loss'),
    ('test', TrainRecordKey.LOSS, 'best_test_loss')
])
def test_train_record_update_smaller(
    train_record, update_type, test_result_key, expected_key
):
    for i in range(10):
        v = 0.1 / (i + 1)
        train_record.update(update_type, {
            test_result_key: v
        })
        train_record.step()
        assert getattr(train_record, update_type)[test_result_key][-1] == v
        assert train_record.best_record[expected_key] == v
        assert train_record.best_record[expected_key + '_epoch'] == i
    
    expected_value = v
    for i in range(10):
        v = i + 1
        train_record.update(update_type, {
            test_result_key: v
        })
        train_record.step()
        assert getattr(train_record, update_type)[test_result_key][-1] == v
        assert train_record.best_record[expected_key] == expected_value
        assert train_record.best_record[expected_key + '_epoch'] == 9

@pytest.mark.parametrize("update_type, test_result_key, expected_key", [
    ('val', TrainRecordKey.ACC, 'best_val_accuracy'),
    ('val', TrainRecordKey.AUC, 'best_val_auc'),
    ('test', TrainRecordKey.ACC, 'best_test_accuracy'),
    ('test', TrainRecordKey.AUC, 'best_test_auc')
])
def test_train_record_update_larger(
    train_record, update_type, test_result_key, expected_key
):
    for i in range(10):
        v = i + 1
        train_record.update(update_type, {
            test_result_key: v
        })
        train_record.step()
        assert getattr(train_record, update_type)[test_result_key][-1] == v
        assert train_record.best_record[expected_key] == v
        assert train_record.best_record[expected_key + '_epoch'] == i
    
    expected_value = v
    for i in range(10):
        v = 0.1 / (i + 1)
        train_record.update(update_type, {
            test_result_key: v
        })
        train_record.step()
        assert getattr(train_record, update_type)[test_result_key][-1] == v
        assert train_record.best_record[expected_key] == expected_value
        assert train_record.best_record[expected_key + '_epoch'] == 9

def test_train_record_update_eval(mocker, train_record):
    update_mock = mocker.patch.object(train_record, 'update')
    train_record.update_eval('testing')
    update_mock.assert_called_once_with('val', 'testing')

def test_train_record_update_test(mocker, train_record):
    update_mock = mocker.patch.object(train_record, 'update')
    train_record.update_test('testing')
    update_mock.assert_called_once_with('test', 'testing')

def test_train_record_update_train(train_record):
    mock = {
        'accuracy': 'testing',
        'loss': 'testing2'
    }
    train_record.update_train(mock)
    for key, value in mock.items():
        assert train_record.train[key][-1] == value

def test_train_record_update_statistic(train_record):
    mock = {
        TrainRecordKey.TIME: 'testing',
        TrainRecordKey.LR: 'testing2'
    }
    train_record.update_statistic(mock)
    for key, value in mock.items():
        assert train_record.train[key][-1] == value

def test_train_record_step(train_record):
    for i in range(10):
        train_record.step()
        assert train_record.get_epoch() == i + 1

@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("val", [True, False])
@pytest.mark.parametrize("test", [True, False])
@pytest.mark.parametrize("test_func, add_target", [
    ('get_loss_figure', TrainRecordKey.LOSS),
    ('get_acc_figure', TrainRecordKey.ACC),
    ('get_auc_figure', TrainRecordKey.AUC)
])
def test_train_record_test_line_figure(
    train_record, train, val, test, test_func, add_target
):
    counter = 0
    if train:
        counter += 1
        train_record.update_train({add_target: 1})
    if val:
        counter += 1
        train_record.update_eval({add_target: 1})
    if test:
        counter += 1
        train_record.update_test({add_target: 1})

    if not train and not val and not test:
        assert getattr(train_record, test_func)() is None
    else:
        figure = getattr(train_record, test_func)()
        assert len(figure.axes[0].lines) == counter
    plt.close('all')

def test_train_record_test_lr_figure(train_record):
    assert train_record.get_lr_figure() is None
    train_record.update_statistic({
        TrainRecordKey.LR: 1
    })
    figure = train_record.get_lr_figure()
    assert len(figure.axes[0].lines) == 1
    plt.close('all')

@pytest.fixture()
def eval_record():
    label = np.arange(CLASS_NUM).repeat(CLASS_NUM)
    output = np.zeros((CLASS_NUM * CLASS_NUM, CLASS_NUM))
    for i in range(CLASS_NUM):
        output[i * CLASS_NUM: (i + 1) * CLASS_NUM, i] = 1
    output[0][1] = 10
    gradient = {}
    return EvalRecord(label, output, gradient)

def test_train_record_test_confusion_figure(train_record, eval_record):
    assert train_record.get_confusion_figure() is None
    train_record.set_eval_record(eval_record)
    figure = train_record.get_confusion_figure()
    assert len(figure.axes[0].images) == 1
    plt.close('all')

@pytest.mark.parametrize("func_name", [
    'get_acc', 'get_auc', 'get_kappa', 'get_eval_record'
])
def test_train_record_eval_record_getter(train_record, eval_record, func_name):
    assert getattr(train_record, func_name)() is None
    train_record.set_eval_record(eval_record)
    assert getattr(train_record, func_name)() is not None

@pytest.mark.parametrize("best_type", ['val', 'test'])
@pytest.mark.parametrize("key", [i for i in RecordKey()])
def test_export(train_record, best_type, key):
    train_record.export_checkpoint()
    
    args_list = torch.save.call_args_list
    files = []
    for args in args_list:
        files.append(os.path.basename(args[0][1]))
    assert 'Epoch-0-model' in files
    assert 'record' in files

    key = 'best_' + best_type + '_' + key + '_model'
    setattr(train_record, key, "test")
    train_record.export_checkpoint()

    args_list = torch.save.call_args_list
    arg = []
    for args in args_list:
        arg.append(args[0][0])
    assert 'test' in arg