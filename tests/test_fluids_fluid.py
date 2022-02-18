"""Test functions for digirock._fluid module
"""
import pytest
from pytest import approx
from _pytest.fixtures import SubRequest

import numpy as np
import xarray as xr
from digirock import Fluid, FluidSwitch
from digirock._exceptions import PrototypeError, WorkflowError
from inspect import getmembers, isfunction


@pytest.fixture
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


def test_Fluid_init():
    assert isinstance(Fluid(), Fluid)


@pytest.mark.parametrize("f", ["density", "velocity", "bulk_modulus"])
def test_Fluid_prototypes(f):
    with pytest.raises(PrototypeError):
        Fluid().__getattribute__(f)(None)


def test_Fluid_shear_modulus():
    assert Fluid().shear_modulus({}) == 0.0


def test_Fluid_get_summary():
    assert isinstance(Fluid().get_summary(), dict)


def test_FluidSwitch():
    f1 = Fluid(name="f1")
    f1.density = lambda props, **kwargs: 11.0
    f1.bulk_modulus = lambda props, **kwargs: 12.0
    f1.velocity = lambda props, **kwargs: 13.0
    f2 = Fluid(name="f2")
    f2.density = lambda props, **kwargs: 21.0
    f2.bulk_modulus = lambda props, **kwargs: 22.0
    f2.velocity = lambda props, **kwargs: 23.0

    fs = FluidSwitch("f", [f1, f2], name="fs")

    assert np.allclose(fs.density({"f": [0, 1]}), np.r_[11, 21])
    assert np.allclose(fs.velocity({"f": [0, 1]}), np.r_[13, 23])
    assert np.allclose(fs.bulk_modulus({"f": [0, 1]}), np.r_[12, 22])
    assert np.allclose(fs.shear_modulus({"f": [0, 1]}), 0)
