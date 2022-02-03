#!/usr/bin/python
# -*- coding: utf8 -*-

"""Test functions for utils.tools module

These test functions are designed to test core functionality with pytest

"""

import digirock.utils as utils
import pytest

from pytest import approx

increasing = list(range(0,10))
decreasing = list(range(0,10))
decreasing.reverse()
test_dict = {i:i for i in 'abcde'}

n = [1, 2, 3, 4]
test_n = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
          [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]

def test_ndim_index_list():
    out = utils.ndim_index_list(n)
    out = [list(a) for a in out]
    assert(out == test_n)

def test_ndim_index_list_ValueError():
    with pytest.raises(ValueError):
        out = utils.ndim_index_list(1)
    with pytest.raises(ValueError):
        out = utils.ndim_index_list('a')

def test_ndim_index_list_TypeError():
    with pytest.raises(TypeError):
        out = utils.ndim_index_list([2.0, 10.0])

if __name__ == "__main__":
    out = utils.ndim_index_list([2.0, 10.5])
    chk = utils.np.all(out == test_n)
    print(chk)