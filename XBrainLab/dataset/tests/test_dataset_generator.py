import numpy as np
import pytest

from XBrainLab.dataset import (
    Dataset,
    DatasetGenerator,
    DataSplitter,
    DataSplittingConfig,
    SplitByType,
    SplitUnit,
    TrainingType,
    ValSplitByType,
)

from .test_epochs import (
    block_size,
    epochs,  # noqa: F401
    n_class,
    n_trial,
    preprocessed_data_list,  # noqa: F401
    session_list,
    subject_list,
)


def test_dataset_generator(
    epochs, # noqa: F811
):
    train_type = TrainingType.IND
    is_cross_validation = False
    test_splitter_list = []
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    with pytest.raises(AssertionError):
        DatasetGenerator(epochs, config, ['test'])
    DatasetGenerator(epochs, config)

@pytest.mark.parametrize('split_type, mock_target', [
    (SplitByType.SESSION, 'pick_session'),
    (SplitByType.SESSION_IND, 'pick_session'),
    (SplitByType.SUBJECT, 'pick_subject'),
    (SplitByType.SUBJECT_IND, 'pick_subject'),
    (SplitByType.TRIAL, 'pick_trial'),
    (SplitByType.TRIAL_IND, 'pick_trial')
])
def test_dataset_generator_split_test(
    epochs, # noqa: F811
    mocker, split_type, mock_target
):
    # prepare generator
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    test_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    test_mask[:3] = True
    expected_next_mask = np.array([False, True, False, True, False])
    split_func_mock = mocker.patch.object(epochs, mock_target,
                                          return_value=(
                                              test_mask,
                                              expected_next_mask
                                          ))

    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    next_mask = generator.split_test(dataset, group_idx, mask, clean_mask)
    call_args = split_func_mock.call_args[1]
    assert np.array_equal(call_args['mask'], mask)
    assert np.array_equal(call_args['clean_mask'], clean_mask)
    assert call_args['value'] == split_value
    assert call_args['split_unit'] == split_unit
    assert call_args['group_idx'] == group_idx

    assert dataset.get_test_len() == 3
    X, y = dataset.get_test_data()
    for i in range(3):
        expected = np.zeros(X[i].shape)
        expected[:, :] = 1 * 100000 + 1 * 1000 + i
        assert np.array_equal(X[i], expected)
        assert y[i] == np.arange(n_class).repeat(n_trial)[i]

    assert (next_mask == expected_next_mask).all()

@pytest.mark.parametrize('split_type, mock_target', [
    (SplitByType.SESSION, 'pick_session'),
    (SplitByType.SESSION_IND, 'pick_session'),
    (SplitByType.SUBJECT, 'pick_subject'),
    (SplitByType.SUBJECT_IND, 'pick_subject'),
    (SplitByType.TRIAL, 'pick_trial'),
    (SplitByType.TRIAL_IND, 'pick_trial')
])
def test_dataset_generator_split_test_list(
    epochs, # noqa: F811
    mocker, split_type, mock_target
):
    # prepare generator
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(split_type, split_value, split_unit),
        DataSplitter(split_type, split_value, split_unit)
    ]
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    test_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    test_mask[:3] = True
    expected_next_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    split_func_mock = mocker.patch.object(epochs, mock_target,
                                          return_value=(
                                              test_mask,
                                              expected_next_mask
                                          ))

    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    generator.split_test(dataset, group_idx, mask, clean_mask)

    call_args_lists = split_func_mock.call_args_list
    call_args = call_args_lists[0][1]
    assert np.array_equal(call_args['mask'], mask)
    assert np.array_equal(call_args['clean_mask'], clean_mask)
    assert call_args['value'] == split_value
    assert call_args['split_unit'] == split_unit
    assert call_args['group_idx'] == group_idx

    call_args = call_args_lists[1][1]
    assert np.array_equal(call_args['mask'], test_mask)
    assert call_args['clean_mask'] is None
    assert call_args['value'] == split_value
    assert call_args['split_unit'] == split_unit
    assert call_args['group_idx'] == group_idx

def test_dataset_generator_split_test_empty(
    epochs, # noqa: F811
    mocker
):
    # prepare generator
    split_type = SplitByType.SESSION
    mock_target = 'pick_session'
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(split_type, split_value, split_unit),
        DataSplitter(split_type, split_value, split_unit)
    ]
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    test_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    expected_next_mask = np.array([False, True, False, True, False])
    split_func_mock = mocker.patch.object(epochs, mock_target,
                                          return_value=(
                                              test_mask,
                                              expected_next_mask
                                          ))

    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    next_mask = generator.split_test(dataset, group_idx, mask, clean_mask)
    call_args = split_func_mock.call_args[1]
    assert np.array_equal(call_args['mask'], mask)
    assert np.array_equal(call_args['clean_mask'], clean_mask)
    assert call_args['value'] == split_value
    assert call_args['split_unit'] == split_unit
    assert call_args['group_idx'] == group_idx

    X, y = dataset.get_test_data()
    assert dataset.get_test_len() == 0
    assert len(X) == 0
    assert len(y) == 0

    assert (~next_mask).all()
    assert not generator.preview_failed

    generator.set_interrupt()
    assert generator.interrupted
    assert generator.preview_failed
    assert not generator.is_clean()

def test_dataset_generator_split_test_failed(
    epochs, # noqa: F811
):
    # prepare generator
    split_type = SplitByType.SESSION
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 30
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)

    with pytest.raises(ValueError):
        generator.split_test(dataset, group_idx, mask, clean_mask)

    assert generator.preview_failed

def test_dataset_generator_split_test_interrupted(
    epochs, # noqa: F811
):
    # prepare generator
    split_type = SplitByType.SESSION
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 30
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)

    generator.set_interrupt()
    with pytest.raises(KeyboardInterrupt):
        generator.split_test(dataset, group_idx, mask, clean_mask)
    assert generator.interrupted
    assert generator.preview_failed


@pytest.mark.parametrize('target_getter_name, expected_length, test_splitter_list', [
    ('get_subject_list', len(subject_list) - 1, [
        DataSplitter(SplitByType.SUBJECT, 1, SplitUnit.NUMBER),
        DataSplitter(SplitByType.SESSION_IND, 1, SplitUnit.NUMBER)
    ]),
    ('get_session_list', len(session_list) - 1, [
        DataSplitter(SplitByType.SESSION, 1, SplitUnit.NUMBER),
        DataSplitter(SplitByType.SUBJECT_IND, 1, SplitUnit.NUMBER)
    ])
])
def test_dataset_generator_split_test_independent(
    epochs, # noqa: F811
    target_getter_name, expected_length, test_splitter_list
):
    train_type = TrainingType.FULL
    is_cross_validation = False
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.ones(epochs.get_data_length(), dtype=bool)
    clean_mask = np.ones(epochs.get_data_length(), dtype=bool)
    generator.split_test(dataset, group_idx, mask, clean_mask)
    assert len(
        np.unique(getattr(epochs, target_getter_name)()[dataset.get_remaining_mask()])
    ) == expected_length

@pytest.mark.parametrize('split_type', ['error', None])
def test_dataset_generator_split_test_not_implemented(
    epochs, # noqa: F811
    split_type
):
    # prepare generator
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    test_splitter_list = [
        DataSplitter(SplitByType.SESSION, split_value, split_unit)
    ]
    test_splitter_list[0].split_type = split_type
    val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    dataset = Dataset(epochs, config)
    group_idx = 0
    mask = np.zeros(epochs.get_data_length(), dtype=bool)
    clean_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    with pytest.raises(NotImplementedError):
        generator.split_test(dataset, group_idx, mask, clean_mask)

@pytest.mark.parametrize('split_type, mock_target', [
    (ValSplitByType.SESSION, 'pick_session'),
    (ValSplitByType.SUBJECT, 'pick_subject'),
    (ValSplitByType.TRIAL, 'pick_trial'),
])
def test_dataset_generator_split_validation(
    epochs, # noqa: F811
    mocker, split_type, mock_target
):
    train_type = TrainingType.FULL
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    val_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    test_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    dataset = Dataset(epochs, config)

    mask = np.ones(epochs.get_data_length(), dtype=bool)

    test_mask = np.zeros(epochs.get_data_length(), dtype=bool)
    test_mask[:3] = True
    expected_next_mask = np.array([False, True, False, True, False])
    split_func_mock = mocker.patch.object(epochs, mock_target,
                                          return_value=(
                                              test_mask,
                                              expected_next_mask
                                          ))

    group_idx = 0
    generator.split_validate(dataset, group_idx)

    call_args = split_func_mock.call_args[1]

    assert np.array_equal(call_args['mask'], mask)
    assert np.array_equal(call_args['clean_mask'], None)
    assert call_args['value'] == split_value
    assert call_args['split_unit'] == split_unit
    assert call_args['group_idx'] == group_idx

    assert dataset.get_val_len() == 3
    X, y = dataset.get_val_data()
    for i in range(3):
        expected = np.zeros(X[i].shape)
        expected[:, :] = 1 * 100000 + 1 * 1000 + i
        assert np.array_equal(X[i], expected)
        assert y[i] == np.arange(n_class).repeat(n_trial)[i]

def test_dataset_generator_split_validation_failed(
    epochs, # noqa: F811
):
    # prepare generator
    split_type = ValSplitByType.SESSION
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 30
    split_unit = SplitUnit.RATIO
    val_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    test_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    dataset = Dataset(epochs, config)
    group_idx = 0

    with pytest.raises(ValueError):
        generator.split_validate(dataset, group_idx)

    assert generator.preview_failed
    with pytest.raises(ValueError):
        generator.generate()
    with pytest.raises(ValueError):
        generator.prepare_reuslt()


@pytest.mark.parametrize('split_type', ['error', None])
def test_dataset_generator_split_validation_not_implemented(
    epochs, # noqa: F811
    split_type
):
    # prepare generator
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 0.2
    split_unit = SplitUnit.RATIO
    test_splitter_list = []
    val_splitter_list = [
        DataSplitter(ValSplitByType.SESSION, split_value, split_unit)
    ]
    val_splitter_list[0].split_type = split_type
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    dataset = Dataset(epochs, config)
    group_idx = 0
    with pytest.raises(NotImplementedError):
        generator.split_validate(dataset, group_idx)

def test_dataset_generator_split_validation_interrupt(
    epochs, # noqa: F811
):
    # prepare generator
    split_type = ValSplitByType.SESSION
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 30
    split_unit = SplitUnit.RATIO
    val_splitter_list = [
        DataSplitter(split_type, split_value, split_unit)
    ]
    test_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    dataset = Dataset(epochs, config)
    group_idx = 0

    generator.set_interrupt()
    with pytest.raises(KeyboardInterrupt):
        generator.split_validate(dataset, group_idx)
    assert generator.interrupted
    assert generator.preview_failed

def test_dataset_generator_failed(
    epochs, # noqa: F811
):
    train_type = TrainingType.FULL
    is_cross_validation = False
    val_splitter_list = []
    test_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    generator.preview_failed = True
    assert not generator.is_clean()

    with pytest.raises(ValueError):
        generator.prepare_reuslt()

    with pytest.raises(ValueError):
        generator.generate()

    generator.reset()
    assert generator.is_clean()

@pytest.mark.parametrize('test_scheme_func_name, expected_name_prefix', [
    ('handle_IND', 'Subject-'),
    ('handle_FULL', 'Group'),
])
def test_dataset_generator_name_prefix(
    epochs, # noqa: F811
    mocker, test_scheme_func_name, expected_name_prefix
):
    train_type = TrainingType.FULL
    is_cross_validation = False
    val_splitter_list = []
    test_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)

    handle_mock = mocker.patch.object(generator, 'handle')
    getattr(generator, test_scheme_func_name)()
    called_args = handle_mock.call_args[0]
    assert called_args[0].startswith(expected_name_prefix)

def test_dataset_generator_handle_individual(
    epochs, # noqa: F811
):
    train_type = TrainingType.IND
    is_cross_validation = False
    split_value = 1
    split_unit = SplitUnit.NUMBER
    test_splitter_list = [
        DataSplitter(SplitByType.SESSION, split_value, split_unit)
    ]
    split_value = 0.25
    split_unit = SplitUnit.RATIO
    val_splitter_list = [
        DataSplitter(ValSplitByType.TRIAL, split_value, split_unit)
    ]
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    result = generator.generate()
    assert len(result) == len(subject_list)
    for i in range(len(result)):
        assert result[i].get_name() == str(i) + '-Subject-' + str(i + 1) + '_0'
        X, _ = result[i].get_training_data()
        assert ((X // 1000 * 1000) == ((i + 1) * 100000 + 2 * 1000)).all()
        X, _ = result[i].get_val_data()
        assert ((X // 1000 * 1000) == ((i + 1) * 100000 + 2 * 1000)).all()
        X, _ = result[i].get_test_data()
        assert ((X // 1000 * 1000) == ((i + 1) * 100000 + 1 * 1000)).all()

def test_dataset_generator_handle_individual_cross_validation(
    epochs, # noqa: F811
):
    train_type = TrainingType.IND
    is_cross_validation = True
    split_value = 1
    split_unit = SplitUnit.NUMBER
    test_splitter_list = [
        DataSplitter(SplitByType.SESSION, split_value, split_unit)
    ]
    split_value = 0.25
    split_unit = SplitUnit.RATIO
    val_splitter_list = [
        DataSplitter(ValSplitByType.TRIAL, split_value, split_unit)
    ]
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    result = generator.generate()
    assert len(result) == len(subject_list) * len(session_list)
    for i in range(len(subject_list)):
        for j in range(len(session_list)):
            idx = i * len(session_list) + j
            assert (
                result[idx].get_name() ==
                str(idx) + '-Subject-' + str(i + 1) + '_' + str(j)
            )
            X, _ = result[idx].get_training_data()
            assert (
                (X // 1000 * 1000) ==
                ((i + 1) * 100000 + ((j + 1) % len(session_list) + 1) * 1000)
            ).all()
            X, _ = result[idx].get_val_data()
            assert (
                (X // 1000 * 1000) ==
                ((i + 1) * 100000 + ((j + 1) % len(session_list) + 1) * 1000)
            ).all()
            X, _ = result[idx].get_test_data()
            assert (
                (X // 1000 * 1000) ==
                ((i + 1) * 100000 + ((j) % len(session_list) + 1) * 1000)
            ).all()

def test_dataset_generator_handle_full(
    epochs, # noqa: F811
):
    train_type = TrainingType.FULL
    is_cross_validation = False
    split_value = 1
    split_unit = SplitUnit.NUMBER
    test_splitter_list = [
        DataSplitter(SplitByType.SUBJECT, split_value, split_unit)
    ]
    split_value = 1
    split_unit = SplitUnit.NUMBER
    val_splitter_list = [
        DataSplitter(ValSplitByType.SESSION, split_value, split_unit)
    ]
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    result = generator.generate()
    assert len(result) == 1
    result = result[0]
    assert result.get_name() == '0-Group_0'
    X, _ = result.get_training_data()
    assert ((X // 100000 * 100000) != (1 * 100000)).all()
    assert len(X) == block_size * ((len(subject_list) - 1) * (len(session_list) - 1))
    X, _ = result.get_val_data()
    assert ((X // 100000 * 100000) != (1 * 100000)).all()
    assert len(X) == block_size * ((len(subject_list) - 1) * 1)
    X, _ = result.get_test_data()
    assert ((X // 100000 * 100000) == (1 * 100000)).all()
    assert len(X) == block_size * len(session_list)


def test_dataset_generator_handle_full_cross_validation(
    epochs, # noqa: F811
):
    train_type = TrainingType.FULL
    is_cross_validation = True
    split_value = 1
    split_unit = SplitUnit.NUMBER
    test_splitter_list = [
        DataSplitter(SplitByType.SUBJECT, split_value, split_unit)
    ]
    split_value = 1
    split_unit = SplitUnit.NUMBER
    val_splitter_list = [
        DataSplitter(ValSplitByType.SESSION, split_value, split_unit)
    ]
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    result = generator.generate()
    assert len(result) == len(subject_list)
    for i in range(len(subject_list)):

        assert result[i].get_name() == str(i) + '-Group_' + str(i)
        X, _ = result[i].get_training_data()
        assert ((X // 100000 * 100000) != ((i + 1) * 100000)).all()
        assert (
            len(X) ==
            block_size * ((len(subject_list) - 1) * (len(session_list) - 1))
        )
        X, _ = result[i].get_val_data()
        assert ((X // 100000 * 100000) != ((i + 1) * 100000)).all()
        assert len(X) == block_size * ((len(subject_list) - 1) * 1)
        X, _ = result[i].get_test_data()
        assert ((X // 100000 * 100000) == ((i + 1) * 100000)).all()
        assert len(X) == block_size * len(session_list)

@pytest.mark.parametrize('train_type, handle_func_name', [
    (TrainingType.IND, 'handle_IND'),
    (TrainingType.FULL, 'handle_FULL')
])
@pytest.mark.parametrize('datasets, has_error', [
    ([], True),
    ([1], False),
    ([1, 2, 3], False)
])
def test_dataset_generator_generate(
    epochs, # noqa: F811
    mocker, train_type, handle_func_name, datasets, has_error
):
    is_cross_validation = False
    test_splitter_list = val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    def handle():
        generator.datasets = datasets
    handle_mock = mocker.patch.object(generator, handle_func_name, side_effect=handle)
    if has_error:
        with pytest.raises(ValueError):
            generator.generate()
        assert not generator.is_clean()
    else:
        generator.generate()
    handle_mock.assert_called_once()

@pytest.mark.parametrize('train_type', ["error", None])
def test_dataset_generator_generate_not_implemented(
    epochs, # noqa: F811
    train_type
):
    is_cross_validation = False
    test_splitter_list = val_splitter_list = []
    config = DataSplittingConfig(
        TrainingType.IND, is_cross_validation, val_splitter_list, test_splitter_list
    )
    config.train_type = train_type
    generator = DatasetGenerator(epochs, config)
    with pytest.raises(NotImplementedError):
        generator.generate()


@pytest.mark.parametrize('train_type', [
    (TrainingType.IND),
    (TrainingType.FULL)
])
def test_dataset_generator_generate_exists(
    epochs, # noqa: F811
    train_type
):
    is_cross_validation = False
    test_splitter_list = val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    datasets = [1, 2, 3]
    generator.datasets = datasets
    result = generator.generate()
    assert result == datasets


def _dataset_generator(selected):
    def func(epoch):
        train_type = TrainingType.IND
        is_cross_validation = False
        test_splitter_list = val_splitter_list = []
        config = DataSplittingConfig(
            train_type, is_cross_validation, val_splitter_list, test_splitter_list
        )
        dataset = Dataset(epoch, config)
        dataset.set_selection(selected)
        return dataset
    return func

@pytest.mark.parametrize('datasets, has_error', [
    ([], True),
    ([_dataset_generator(False)], True),
    ([_dataset_generator(True)], False),
    ([_dataset_generator(False), _dataset_generator(True)], False),
    ([_dataset_generator(False), _dataset_generator(False)], True),
])
@pytest.mark.parametrize('train_type', [
    (TrainingType.IND),
    (TrainingType.FULL)
])
def test_dataset_generator_prepare_reuslt(
    epochs, # noqa: F811
    mocker, train_type, datasets, has_error
):
    is_cross_validation = False
    test_splitter_list = val_splitter_list = []
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    for i in range(len(datasets)):
        if not isinstance(datasets[i], Dataset):
            datasets[i] = datasets[i](epochs)
    generator.datasets = datasets
    mocker.patch.object(generator, 'generate')
    if has_error:
        with pytest.raises(ValueError):
            generator.prepare_reuslt()
    else:
        generator.prepare_reuslt()
        assert generator.is_clean()

def test_dataset_generator_apply(epochs): # noqa: F811
    from XBrainLab import Study
    study = Study()
    is_cross_validation = False
    test_splitter_list = val_splitter_list = []
    config = DataSplittingConfig(
        TrainingType.IND, is_cross_validation, val_splitter_list, test_splitter_list
    )
    generator = DatasetGenerator(epochs, config)
    generator.datasets = [Dataset(epochs, config)]
    generator.apply(study)
    assert study.datasets == generator.datasets
