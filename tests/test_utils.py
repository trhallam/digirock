#!/usr/bin/python
# -*- coding: utf8 -*-

"""Test functions for utils.tools module

These test functions are designed to test core functionality with pytest

"""

import pytest
from pytest import approx

import numpy as np

from digirock.utils import ndim_index_list, check_broadcastable, safe_divide


@pytest.fixture(scope="module")
def ndim_index_list_data():
    n = [1, 2, 3, 4]
    test_n = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    ]
    return n, test_n


def test_ndim_index_list(ndim_index_list_data):
    n, test_n = ndim_index_list_data
    out = ndim_index_list(n)
    out = [list(a) for a in out]
    assert out == test_n


def test_ndim_index_list_ValueError():
    with pytest.raises(ValueError):
        out = ndim_index_list(1)
    with pytest.raises(ValueError):
        out = ndim_index_list("a")


def test_ndim_index_list_TypeError():
    with pytest.raises(TypeError):
        out = ndim_index_list([2.0, 10.0])


def test_check_broadcastable_ok():
    assert check_broadcastable(a=1, b=2) == (1,)
    assert check_broadcastable(a=1, b=2, c=None) == (1,)
    assert check_broadcastable(a=1, b=np.r_[1, 2]) == (2,)
    assert check_broadcastable(a=np.r_[1, 2], b=np.zeros((2, 2))) == (2, 2)
    assert check_broadcastable(a=1, b=1, c=1) == (1,)
    assert check_broadcastable(a=1, b=np.r_[1, 2], c=np.zeros((2, 2))) == (2, 2)
    assert check_broadcastable(a=np.r_[1, 2], b=np.zeros((2, 2)), c=1) == (2, 2)


def test_check_broadcastable_fail():
    with pytest.raises(ValueError):
        assert check_broadcastable(a=np.zeros((3, 3)), b=np.r_[1, 2])
    with pytest.raises(ValueError):
        assert check_broadcastable(a=np.r_[1, 2], b=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        assert check_broadcastable(a=1, b=np.zeros((2, 2)), c=np.zeros(10))
    with pytest.raises(ValueError):
        assert check_broadcastable(a=1, b=np.r_[1, 2, 3], c=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        assert check_broadcastable(a=np.r_[1, 2], b=np.zeros((2, 3)), c=1)


def test_safe_divide():
    assert np.allclose(safe_divide(np.arange(10), np.zeros(10)), 0.0)
    assert np.allclose(safe_divide(np.arange(10), 0.0), 0.0)
    assert safe_divide(1.0, 0.0) == 0.0
