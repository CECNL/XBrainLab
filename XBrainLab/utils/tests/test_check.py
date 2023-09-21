import numpy as np
import pytest

from XBrainLab.utils import check


def test__get_type_name():
    assert check._get_type_name(int) == 'builtins.int'
    assert check._get_type_name(np.ndarray) == 'numpy.ndarray'


def test_validate_type():
    check.validate_type(1, int, 'test')
    check.validate_type(1, (float, int), 'test')

    with pytest.raises(
        TypeError,
        match = (
            'test must be an instance of builtins.float, '
            'got <class \'int\'> instead.'
        )
    ):
        check.validate_type(1, float, 'test')
    with pytest.raises(
        TypeError,
        match=(
            'test must be an instance of builtins.float or builtins.int, '
            'got <class \'str\'> instead.'
        )
    ):
        check.validate_type('1', (float, int), 'test')

def test_validate_list_type():
    check.validate_list_type([], int, 'test')
    check.validate_list_type([1, 2, 3], int, 'test')
    check.validate_list_type([1, 2, 3], (float, int), 'test')

    with pytest.raises(
        TypeError,
        match=(
            'Items of test must be an instance of builtins.float, '
            'got <class \'int\'> instead.'
        )
    ):
        check.validate_list_type([1, 2, 3], float, 'test')

    with pytest.raises(
        TypeError,
        match=(
            'Items of test must be an instance of builtins.float or builtins.int, '
            'got <class \'str\'> instead.'
        )
    ):
        check.validate_list_type([1, 2, '3'], (float, int), 'test')

class A:
    pass

class B(A):
    pass

def test_validate_issubclass():
    check.validate_issubclass(B, A, 'test')
    check.validate_issubclass(B, (A, int), 'test')
    answer = "XBrainLab.utils.tests.test_check.B"
    with pytest.raises(
        TypeError,
        match=f'test must be an instance of builtins.int, got {answer} instead.'
    ):
        check.validate_issubclass(B, int, 'test')
    with pytest.raises(
        TypeError,
        match=(
            f'test must be an instance of builtins.int or builtins.float, '
            f'got {answer} instead.'
        )
    ):
        check.validate_issubclass(B, (int, float), 'test')

