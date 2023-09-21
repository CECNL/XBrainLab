import pytest

from XBrainLab.dataset import (
    DataSplitter,
    DataSplittingConfig,
    SplitByType,
    SplitUnit,
    TrainingType,
    ValSplitByType,
)


@pytest.mark.parametrize(
    'split_type', list(SplitByType) + list(ValSplitByType)
)
@pytest.mark.parametrize('parsed_value, value_var, value_target_unit', [
    (0, '0', (SplitUnit.RATIO, SplitUnit.NUMBER, SplitUnit.MANUAL)),
    (0.0, '0.0', (SplitUnit.RATIO,)),
    (0.3, '0.3', (SplitUnit.RATIO,)),
    (1, '1', (SplitUnit.RATIO, SplitUnit.NUMBER, SplitUnit.KFOLD, SplitUnit.MANUAL)),
    (1.0, '1.0', (SplitUnit.RATIO,)),
    (None, '1.5', ()),
    (2, '2', (SplitUnit.NUMBER, SplitUnit.KFOLD, SplitUnit.MANUAL)),
    (None, '2.0', ()),
    (None, '2.0 ', ()),
    ([2], '2 ', (SplitUnit.MANUAL,)),
    (None, '-1', ()),
    (None, '-1 ', ()),
    (None, '-1.0', ()),
    (None, '-1.0 ', ()),
    (None, '-1.5', ()),
    (None, '-1.5 ', ()),
    ([1, 2, 3], '1 2 3', (SplitUnit.MANUAL,)),
    (None, '1 2 -3', ()),
    (None, '1 2 0.3', ()),
    (None, 'e', ()),
    (None, 'e ', ()),
    (None, '1 e', ()),
    (None, None, ()),
])
@pytest.mark.parametrize('split_unit', [*list(SplitUnit), None])
def test_splitter(split_type, parsed_value, value_var, value_target_unit, split_unit):
    is_option = True
    splitter = DataSplitter(split_type, value_var, split_unit, is_option)

    assert splitter.is_option == is_option
    assert splitter.split_type == split_type
    assert splitter.text == split_type.value
    assert splitter.value_var == value_var
    assert splitter.split_unit == split_unit

    if split_unit is None:
        assert not splitter.is_valid()
    else:
        assert splitter.is_valid() == (split_unit in value_target_unit)

    if not splitter.is_valid():
        with pytest.raises(ValueError):
            splitter.get_value()
        with pytest.raises(ValueError):
            splitter.get_raw_value()
    else:
        checker = parsed_value
        if split_unit == SplitUnit.MANUAL and not isinstance(parsed_value, list):
            checker = [parsed_value]
        assert splitter.get_value() == checker
        assert splitter.get_raw_value() == value_var

@pytest.mark.parametrize('split_unit', [*list(SplitUnit), 'test'])
def test_splitter_not_implemented(split_unit):
    split_type = SplitByType.SESSION
    value_var = '1'
    is_option = True
    splitter = DataSplitter(split_type, value_var, SplitUnit.MANUAL, is_option)
    splitter.split_unit = split_unit
    if split_unit == "test":
        with pytest.raises(NotImplementedError):
            splitter.is_valid()
    else:
        splitter.is_valid()

def test_splitter_getter():
    split_type = SplitByType.SESSION
    split_unit = SplitUnit.KFOLD
    value_var = '1'
    is_option = True
    splitter = DataSplitter(split_type, value_var, split_unit, is_option)

    assert splitter.get_split_unit() == split_unit
    assert splitter.get_split_type_repr() == 'SplitByType.SESSION'
    assert splitter.get_split_unit_repr() == 'SplitUnit.KFOLD'



def test_config():
    train_type =  TrainingType.FULL
    is_cross_validation = True
    val_splitter_list = [DataSplitter(SplitByType.SESSION, '1', SplitUnit.KFOLD, True)]
    test_splitter_list = [DataSplitter(SplitByType.SESSION, '1', SplitUnit.KFOLD, True)]
    config = DataSplittingConfig(
        train_type, is_cross_validation, val_splitter_list, test_splitter_list
    )

    assert config.train_type == train_type
    assert config.is_cross_validation == is_cross_validation
    assert config.val_splitter_list == val_splitter_list
    assert config.test_splitter_list == test_splitter_list

    assert config.get_splitter_option() == (val_splitter_list, test_splitter_list)
    assert config.get_train_type_repr() == 'TrainingType.FULL'
