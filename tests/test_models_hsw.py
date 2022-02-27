"""Test functions for modes._hsw module
"""

import pytest
from pytest import approx
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as stn

import numpy as np
import digirock.models as mod
from digirock.utils import check_broadcastable

from .strategies import n_varshp_arrays


@pytest.fixture(scope="module")
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


# k1, k2, mu1, f1
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 20, 15, 1.0), 10.0),
        ((10, 20, 15, 0.0), 20),
        ((10, 20, 15, 0.5), 15),
        ((10, 10, 15, 0.5), 10),
        ((10, 20, 15, 0.3), 16.36),
    ),
)
@given(data=st.data())
def test_hs_kbounds2(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.hs_kbounds2(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# k1, mu1, mu2, f1
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 15, 17, 1.0), 15.0),
        ((10, 15, 17, 0.0), 17),
        ((10, 15, 17, 0.5), 15.96),
        ((10, 15, 15, 0.5), 15),
        ((10, 15, 17, 0.3), 16.36),
    ),
)
@given(data=st.data())
def test_hs_mubounds2(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.hs_mubounds2(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# mod, frac, *argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 1.0, 20), (10.0, 10.0)),
        ((10, 0.0, 20), (20, 20.0)),
        ((10, 0.5, 20, 0.5), (10.0, 20.0)),
        ((10, 0.3, 20, 0.3, 30), (10.0, 30.0)),
        ((10, 0.3, 20, 0.7, 30), (10.0, 20.0)),
    ),
)
@given(data=st.data())
def test_hsw_get_minmax(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test_min = mod._hsw._hsw_get_minmax(*argv, minmax="min")
    test_max = mod._hsw._hsw_get_minmax(*argv, minmax="max")
    assert np.allclose(test_min, ans[0], rtol=tol["rel"])
    assert np.allclose(test_max, ans[1], rtol=tol["rel"])
    assert test_min.shape == result_shape
    assert test_max.shape == result_shape


# z, mod, frac, argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((3, 10, 1.0, 20), 10.0),
        ((3, 10, 0.0, 20), 20),
        ((3, 10, 0.5, 20, 0.5), 13.68),
        ((3, 10, 0.3, 20, 0.3, 30), 17.88),
    ),
)
@given(data=st.data())
def test_hsw_bulk_modulus_avg(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod._hsw._hsw_bulk_modulus_avg(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# z, mod, frac, argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((3, 10, 1.0, 20), 10.0),
        ((3, 10, 0.0, 20), 20),
        ((3, 10, 0.5, 20, 0.5), 13.61),
        ((3, 10, 0.3, 20, 0.3, 30), 17.72),
    ),
)
@given(data=st.data())
def test_hsw_shear_modulus_avg(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod._hsw._hsw_shear_modulus_avg(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# k, mu
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 20), 16.6),
        ((15, 20), 17.87),
        ((12, 40), 31.01),
        ((10, 30), 23.57),
    ),
)
@given(data=st.data())
def test_hsw_zeta(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod._hsw._hsw_zeta(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# k, mu, frac, argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 20, 1.0), (10.0, 10.0, 20.0, 20.0)),
        ((10, 20, 0.5, 20, 15, 0.5), (14.4, 14.29, 17.33, 17.3)),
        ((10, 20, 0.3, 20, 15, 0.3, 30, 45), (20.13, 19.22, 25.94, 24.46)),
    ),
)
@given(data=st.data())
def test_hsw_bounds(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.hsw_bounds(*argv)
    for v, u in zip(ans, test):
        assert np.allclose(v, u, rtol=tol["rel"])
        assert u.shape == result_shape


# k, mu, frac, argv
# fmt: off
@pytest.mark.parametrize("args,ans",(((10, 20, 0.0,), (10.0, 10.0, 20, 20),),),)
# fmt: on
@given(data=st.data())
def test_hsw_bounds_fail(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    with pytest.raises(ValueError):
        assert mod.hsw_bounds(*argv)


# k, mu, frac, argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 20, 1.0), (10.0, 20.0, 20.0, 20.0)),
        ((10, 20, 0.5, 20, 15, 0.5), (14.4, 17.31, 17.33, 17.3)),
        ((10, 20, 0.3, 20, 15, 0.3, 30, 45), (20.13, 25.2, 25.94, 24.46)),
    ),
)
@given(data=st.data())
def test_hsw_avg(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.hsw_avg(*argv)
    for v, u in zip(ans, test):
        assert np.allclose(v, u, rtol=tol["rel"])
        assert u.shape == result_shape
