"""Test functions for modes._mod module
"""

# pylint: disable=invalid-name,redefined-outer-name,missing-docstring

import pytest
from pytest import approx
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as stn

import numpy as np
import digirock.models as mod

from .strategies import n_varshp_arrays


@pytest.fixture(scope="module")
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


# mod, frac, *argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 1.0, 20), 10.0),
        ((10, 0.0, 20), 20),
        ((10, 0.5, 20, 0.5), 15),
        ((10, 0.3, 20, 0.3, 30), 21),
    ),
)
@given(data=st.data())
def test_voight_upper_bound(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.voigt_upper_bound(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# mod, frac, *argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 1.0, 20), 10.0),
        ((10, 0.0, 20), 20),
        ((10, 0.5, 20, 0.5), 13.3),
        ((10, 0.3, 20, 0.3, 30), 17.14),
    ),
)
@given(data=st.data())
def test_reuss_lower_bound(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.reuss_lower_bound(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# mod, frac, *argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 1.0, 20), 10.0),
        ((10, 0.0, 20), 20),
        ((10, 0.5, 20, 0.5), 14.16),
        ((10, 0.3, 20, 0.3, 30), 19.07),
    ),
)
@given(data=st.data())
def test_vrh_avg(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.vrh_avg(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# mod, frac, *argv
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10, 1.0, 20), 10.0),
        ((10, 0.0, 20), 20),
        ((10, 0.5, 20, 0.5), 15),
        ((10, 0.3, 20, 0.3, 30), 21),
    ),
)
@given(data=st.data())
def test_mixed_density(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.mixed_density(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# # mod, frac, *argv
# @pytest.mark.parametrize(
#     "args,ans",
#     (
#         ((10, 1.0, 20), 10.0),
#         ((10, 0.0, 20), 20),
#         ((10, 0.5, 20, 0.5), 15),
#         ((10, 0.3, 20, 0.3, 30), 21),
#     ),
# )
# @given(data=st.data())
# def test_dryframe_dpres(args, ans, data, tol):
#     argv, result_shape = data.draw(n_varshp_arrays(args))
#     test = mod.dryframe_dpres(*argv)
#     assert np.allclose(test, ans, rtol=tol["rel"])
#     assert test.shape == result_shape

#     )


# kdry, kfl, k0, phi
@pytest.mark.parametrize(
    "args,ans",
    (((22, 2, 30, 0.25), 22.56),),
)
@given(data=st.data())
def test_gassmann_fluidsub(args, ans, data, tol):
    argv, result_shape = data.draw(n_varshp_arrays(args))
    test = mod.gassmann_fluidsub(*argv)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape
