from XBrainLab.training import TrainingPlanHolder, Trainer

import pytest

class FakePlan:
    def __init__(self, i):
        self.i = i
    def get_name(self):
        return str(self.i)

class FakeTrainingPlanHolder(TrainingPlanHolder):
    def __init__(self, i):
        self.i = i
        self.train_record_list = [FakePlan('test')]

    def get_name(self):
        return 'Fake' + str(self.i)

@pytest.fixture
def training_plan_holders():
    result = []
    for i in range(2):
        result.append(FakeTrainingPlanHolder(i))
    return result

def test_trainer(mocker, training_plan_holders):
    trainer = Trainer(training_plan_holders)
    assert trainer.get_training_plan_holders() == training_plan_holders
    assert trainer.get_progress_text() == 'Pending'
    holder = training_plan_holders[-1]
    interrupt_mock = mocker.patch.object(holder, 'set_interrupt')
    def interrupt():
        assert trainer.get_progress_text() == 'Interrupting'
    interrupt_mock.side_effect = interrupt
    
    clear_interrupt_mock = mocker.patch.object(holder, 'clear_interrupt')
    
    trainer.set_interrupt()
    interrupt_mock.assert_called_once()
    assert trainer.interrupt
    assert trainer.get_progress_text() == 'Interrupting'

    trainer.clear_interrupt()
    clear_interrupt_mock.assert_called_once()
    assert trainer.interrupt is False
    assert trainer.get_progress_text() == 'Pending'
    
def test_trainer_custom_progress_text(training_plan_holders):
    trainer = Trainer(training_plan_holders)
    trainer.progress_text = 'Custom'
    assert trainer.get_progress_text() == 'Custom'

@pytest.mark.parametrize('interact', [True, False])
def test_trainer_run(mocker, training_plan_holders, interact):
    trainer = Trainer(training_plan_holders)
    job_mock = mocker.patch.object(trainer, 'job')
    def job():
        import threading
        if interact:
            assert trainer.is_running()
            assert not isinstance(threading.current_thread(), threading._MainThread)
            call_count = job_mock.call_count
            trainer.run()
            assert job_mock.call_count == call_count
            with pytest.raises(RuntimeError):
                trainer.clean()
            trainer.clean(force_update=True)
        else:
            assert isinstance(threading.current_thread(), threading._MainThread)
        
    job_mock.side_effect = job
    trainer.run(interact=interact)
    job_mock.assert_called_once()
    assert trainer.is_running() is False

def test_trainer_job(mocker, training_plan_holders):
    trainer = Trainer(training_plan_holders)
    train_mock_list = []
    counter = 0
    def train():
        nonlocal counter
        assert trainer.get_progress_text() == 'Now training: Fake' + str(counter)
        counter += 1
    for holder in training_plan_holders:
        train_mock = mocker.patch.object(holder, 'train')
        train_mock.side_effect = train
        train_mock_list.append(train_mock)
    trainer.job()
    for train_mock in train_mock_list:
        train_mock.assert_called_once()
    assert trainer.get_progress_text() == 'Pending'
    assert trainer.is_running() is False

def test_trainer_interrupt(mocker, training_plan_holders):
    trainer = Trainer(training_plan_holders)
    holder = training_plan_holders[0]
    train_mock = mocker.patch.object(holder, 'train')
    trainer.set_interrupt()
    trainer.job()
    train_mock.assert_not_called()

@pytest.mark.parametrize(
    'plan_name, real_plan_name, error_stage',
    [
        ['Fake', 'test', 1],
        ['Fake0', 'test', 0],
        ['Fake1', 'test', 0],
        ['Fake1', 'tests', 2]
    ]
)
def test_trainer_get_plan(
    training_plan_holders, plan_name, real_plan_name, error_stage
):
    trainer = Trainer(training_plan_holders)
    if error_stage == 0:
        trainer.get_real_training_plan(plan_name, real_plan_name)
    else:
        if error_stage == 1:
            error = '.*training plan.*'
        else:
            error = '.*real plan.*'
        with pytest.raises(ValueError, match=error):
            trainer.get_real_training_plan(plan_name, real_plan_name)