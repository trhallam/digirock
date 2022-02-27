#!/usr/bin/python
# -*- coding: utf8 -*-

"""Test functions for utils.tools module

These test functions are designed to test core functionality with pytest

"""

import pytest
from pytest import approx
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as stn
from .strategies import np_ints_or_floats

import numpy as np

from digirock.utils import ndim_index_list, check_broadcastable, safe_divide
from digirock.utils._utils import _process_vfrac, nan_divide


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


def test_nan_divide():
    assert np.all(np.isnan(nan_divide(np.arange(10), np.zeros(10))))
    assert np.all(np.isnan(nan_divide(np.arange(10), 0.0)))
    assert np.all(np.isnan(nan_divide(1.0, 0.0)))


@given(shps=stn.mutually_broadcastable_shapes(num_shapes=6, min_dims=1, max_dims=4))
def test_proces_vfrac_all_args(shps):
    shps = list(shps.input_shapes)
    argv = (
        np.full(shps[0], 1.0),
        np.full(shps[1], 0.3),
        np.full(shps[2], 1.0),
        np.full(shps[3], 0.3),
        np.full(shps[4], 1.0),
        np.full(shps[5], 0.4),
    )
    test = _process_vfrac(*argv)
    assert len(test) == len(argv)
    for v, u in zip(test, argv):
        assert np.allclose(v, u)


@given(shps=stn.mutually_broadcastable_shapes(num_shapes=6, min_dims=1, max_dims=4))
def test_proces_vfrac_comp_args(shps):
    shps = list(shps.input_shapes)
    comp_shp = np.broadcast_shapes(shps[1], shps[3])
    argv = (
        np.full(shps[0], 1.0),
        np.full(shps[1], 0.3),
        np.full(shps[2], 1.0),
        np.full(shps[3], 0.3),
        np.full(shps[4], 1.0),
        np.full(comp_shp, 0.4),
    )
    test = _process_vfrac(*argv[:-1])
    assert len(test) == len(argv)
    for v, u in zip(test, argv):
        assert np.allclose(v, u)


@given(shps=stn.mutually_broadcastable_shapes(num_shapes=6, min_dims=1, max_dims=4))
def test_proces_vfrac_bad_vfrac(shps):
    shps = list(shps.input_shapes)
    comp_shp = np.broadcast_shapes(shps[1], shps[3])
    argv = (
        np.full(shps[0], 1.0),
        np.full(shps[1], 0.4),
        np.full(shps[2], 1.0),
        np.full(shps[3], 0.3),
        np.full(shps[4], 1.0),
        np.full(comp_shp, 0.4),
    )
    with pytest.raises(ValueError):
        assert _process_vfrac(*argv)


@given(shps=stn.mutually_broadcastable_shapes(num_shapes=6, min_dims=1, max_dims=4))
def test_process_vfrac_n_args(shps):
    shps = list(shps.input_shapes)
    comp_shp = np.broadcast_shapes(shps[1], shps[3])
    argv = (
        np.full(shps[0], 1.0),
        np.full(shps[1], 2.4),
        np.full(shps[2], 0.6),
        np.full(shps[3], 2.3),
        np.full(shps[4], 3.0),
        np.full(comp_shp, 0.4),
    )

    test = _process_vfrac(*argv, i=2)
    assert len(test) == len(argv)
    for v, u in zip(test, argv):
        assert np.allclose(v, u)

    test = _process_vfrac(*argv[:-1], i=2)
    assert len(test) == len(argv)
    for v, u in zip(test, argv):
        assert np.allclose(v, u)

    with pytest.raises(ValueError):
        assert _process_vfrac(*argv[:-2], i=2)
