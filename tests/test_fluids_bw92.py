"""Test functions for pem.fluid.bw92 module
"""
import pathlib

import pytest
from pytest import approx
from _pytest.fixtures import SubRequest

import numpy as np
import digirock.fluids.bw92 as bw92
import digirock.fluids.ecl as fluid_ecl
from inspect import getmembers, isfunction


@pytest.fixture
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


@pytest.fixture
def dummy_values(request: SubRequest):
    param = getattr(request, "param", None)
    keys = param[:-1]
    ans = param[-1]
    values = {
        "tk": 373,  # temperature in kelvin
        "t": 273.15,  # temperature in degc
        "p": 10 * 1e6,  # pressure in pascals
        "orho": 0.8,  # oil density g/cc
        "M": 16.04,  # methane gas molecular weight
        "G": 0.56,  # methane specific gravity
        "ta": np.array([273.15, 273.15 + 100]),  # temperature array in kelvin
        "pa": np.array([10 * 1e6, 50 * 1e6]),  # pressure array in pascals
        "orho_a": np.array([0.8, 0.9]),
        "rg": 120,  # solution gas
        "sal": 32000,
        "sal_a": np.r_[32000, 150000],
        "vel": 1300,
        "vel_a": np.r_[1300, 1450],
        "wrho": 1.0,
        "wrho_a": np.r_[1.0, 1.1],
    }
    if keys:
        v = tuple(values[k] for k in keys)
    else:
        v = tuple(values.values)
    yield v, ans


def test_GAS_R():
    assert bw92.GAS_R == 8.31441


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "t", 0.000454216),
        ("pa", "t", (0.000454216, 9.08432e-05)),
        ("p", "ta", (0.000454216, 0.00053736)),
        ("pa", "ta", (0.000454216, 0.00010747)),
    ],
    indirect=True,
)
def test_gas_vmol(dummy_values, tol):
    param, ans = dummy_values
    p, t = param
    assert np.allclose(bw92.gas_vmol(t, p), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "t", "M", 35313.5783218),
        ("pa", "t", "M", (35313.5783218, 176567.89160)),
        ("p", "ta", "M", (35313.5783218, 29849.6175727)),
        ("pa", "ta", "M", (35313.5783218, 149248.08786351)),
    ],
    indirect=True,
)
def test_gas_density(dummy_values, tol):
    param, ans = dummy_values
    p, t, M = param
    assert np.allclose(bw92.gas_density(M, t, p), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("pa", "t", 2e-08),
    ],
    indirect=True,
)
def test_gas_isotherm_comp(dummy_values, tol):
    param, ans = dummy_values
    pa, t = param
    v1, v2 = bw92.gas_vmol(t, pa)
    assert bw92.gas_isotherm_comp(v1, v2, pa[0], pa[1]) == approx(ans)


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("M", "t", 16.8278695),
        ("M", "ta", (16.82786955, 18.30335126)),
    ],
    indirect=True,
)
def test_gas_isotherm_vp(dummy_values, tol):
    param, ans = dummy_values
    M, t = param
    assert np.allclose(bw92.gas_isotherm_vp(M, t), ans, rtol=tol["rel"])


@pytest.mark.parametrize("dummy_values", [("G", 4.665312)], indirect=True)
def test_gas_pseudocrit_pres(dummy_values):
    G, ans = dummy_values
    G = np.array(G)
    assert bw92.gas_pseudocrit_pres(G) == approx(ans)


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "G", 2143479.36429546),
        ("pa", "G", (2143479.36429546, 10717396.82147732)),
    ],
    indirect=True,
)
def test_gas_pseudored_pres(dummy_values, tol):
    param, ans = dummy_values
    p, G = param
    assert np.allclose(bw92.gas_pseudored_pres(p, G), ans, rtol=tol["rel"])


@pytest.mark.parametrize("dummy_values", [("G", 190.34)], indirect=True)
def test_gas_pseudocrit_temp(dummy_values):
    G, ans = dummy_values
    G = np.array(G)
    assert bw92.gas_pseudocrit_temp(G) == approx(ans)


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("t", "G", 2.87012714),
        ("ta", "G", (2.87012714, 3.39550278)),
    ],
    indirect=True,
)
def test_gas_pseudored_temp(dummy_values, tol):
    param, ans = dummy_values
    t, G = param
    assert np.allclose(bw92.gas_pseudored_temp(t, G), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "t", "G", 0.5289487894),
        ("pa", "t", "G", (0.5289487894, 0.52895413)),
        ("p", "ta", "G", (0.5289487894, 0.46664469)),
        ("pa", "ta", "G", (0.5289487894, 0.46664469)),
    ],
    indirect=True,
)
def test_gas_oga_density(dummy_values, tol):
    param, ans = dummy_values
    p, t, G = param
    assert bw92.gas_oga_density(t, p, G) == approx(ans, rel=tol["rel"])


def test_gas_oga_density_warning():
    with pytest.warns(UserWarning):
        bw92.gas_oga_density(4.5, 4.5, 1)


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "t", "G", 673174274.6197122),
        ("pa", "t", "G", (673174274.6197122, 1.68291455e10)),
        ("p", "ta", "G", (6.73174275e08, 7.49509086e08)),
        ("pa", "ta", "G", (6.73174275e08, 1.87375111e10)),
    ],
    indirect=True,
)
def test_gas_adiabatic_bulkmod(dummy_values, tol):
    param, ans = dummy_values
    p, t, G = param
    assert bw92.gas_adiabatic_bulkmod(t, p, G) == approx(ans, rel=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "t", "G", 0.0204339351378),
        ("pa", "t", "G", (0.0204339351378, 0.02968152)),
        ("p", "ta", "G", (0.0204339351378, 0.02277815)),
        ("pa", "ta", "G", (0.0204339351378, 0.03011878)),
    ],
    indirect=True,
)
def test_gas_adiabatic_viscosity(dummy_values, tol):
    param, ans = dummy_values
    p, t, G = param
    assert bw92.gas_adiabatic_viscosity(t, p / 1e6, G) == approx(ans, rel=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "orho", 0.8068623025),
        ("pa", "orho", (0.8068623025, 0.83179781)),
        ("pa", "orho_a", (0.8068623, 0.92477031)),
        ("p", "orho_a", (0.8068623, 0.90521056)),
    ],
    indirect=True,
)
def test_oil_isothermal_density(dummy_values, tol):
    param, ans = dummy_values
    p, rho = param
    assert np.allclose(bw92.oil_isothermal_density(rho, p / 1e6), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("p", "orho", "t", 0.63475419),
        ("pa", "orho", "t", (0.63475419, 0.65437082)),
        ("pa", "orho_a", "t", (0.63475419, 0.72751178)),
        ("p", "orho_a", "t", (0.63475419, 0.71212423)),
        ("p", "orho", "ta", (0.63475419, 0.57827437)),
        ("pa", "orho", "ta", (0.63475419, 0.59614553)),
        ("pa", "orho_a", "ta", (0.63475419, 0.66277848)),
        ("p", "orho_a", "ta", (0.63475419, 0.66277848)),
    ],
    indirect=True,
)
def test_oil_density(dummy_values, tol):
    param, ans = dummy_values
    p, rho, t = param
    assert np.allclose(bw92.oil_density(rho, p / 1e6, t), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values", [("orho", "G", "rg", "t", 1.57823582)], indirect=True
)
def test_oil_fvf(dummy_values, tol):
    param, ans = dummy_values
    rho, G, rg, t = param
    assert np.allclose(bw92.oil_fvf(rho, G, rg, t), ans, rtol=tol["rel"])


# @pytest.mark.parametrize('')
# def test_oil_rg(oil, g, pres, temp, mode='rho'):
#     param, ans = dummy_values
#     p, t, G = param

# def test_oil_rho_sat(rho0, g, rg, b0):
#     param, ans = dummy_values
#     p, t, G = param

# def test_oil_rho_pseudo(rho0, rg, b0):
#     param, ans = dummy_values
#     p, t, G = param

# def test_oil_velocity(rho0, p, t, g, rg, b0=None):
#     param, ans = dummy_values
#     p, t, G = param

# def test_oil_bulkmod(rho, vp):
#     param, ans = dummy_values
#     p, t, G = param


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("t", "p", 1063.23709),
        ("t", "pa", (1063.23709, 1272.157219)),
        ("ta", "p", (1063.23709047, 202.38059444)),
        ("ta", "pa", (1063.23709047, 847.72401465)),
    ],
    indirect=True,
)
def test_wat_velocity_pure(dummy_values, tol):
    param, ans = dummy_values
    t, p = param
    assert np.allclose(bw92.wat_velocity_pure(t, p / 1e6), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("t", "p", "sal", 1095.70072),
        ("t", "pa", "sal", (1095.70072588, 1305.21551832)),
        ("ta", "p", "sal", (1095.70072588, 232.07427709)),
        ("ta", "pa", "sal", (1095.70072588, 878.38356076)),
        ("t", "p", "sal_a", (1095.70072588, 1204.95468785)),
        ("t", "pa", "sal_a", (1095.70072588, 1416.16211083)),
        ("ta", "p", "sal_a", (1095.70072588, 331.11403825)),
        ("ta", "pa", "sal_a", (1095.70072588, 980.48475247)),
    ],
    indirect=True,
)
def test_wat_velocity_brine(dummy_values, tol):
    param, ans = dummy_values
    t, p, sal = param
    assert np.allclose(
        bw92.wat_velocity_brine(t, p / 1e6, sal / 1e6), ans, rtol=tol["rel"]
    )


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("t", "p", 0.776224),
        ("t", "pa", (0.776224, 0.808975)),
        ("ta", "p", (0.776224, 0.614422)),
        ("ta", "pa", (0.776224, 0.663635)),
    ],
    indirect=True,
)
def test_wat_density_pure(dummy_values, tol):
    param, ans = dummy_values
    t, p = param
    assert np.allclose(bw92.wat_density_pure(t, p / 1e6), ans, rtol=tol["rel"])


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("t", "p", "sal", 0.804056),
        ("t", "pa", "sal", (0.804056, 0.83307377)),
        ("ta", "p", "sal", (0.804056, 0.64800884)),
        ("ta", "pa", "sal", (0.804056, 0.69201752)),
        ("t", "p", "sal_a", (0.804056, 0.90036779)),
        ("t", "pa", "sal_a", (0.804056, 0.92300742)),
        ("ta", "p", "sal_a", (0.804056, 0.76053159)),
        ("ta", "pa", "sal_a", (0.804056, 0.79606398)),
    ],
    indirect=True,
)
def test_wat_density_brine(dummy_values, tol):
    param, ans = dummy_values
    t, p, sal = param
    assert np.allclose(
        bw92.wat_density_brine(t, p / 1e6, sal / 1e6), ans, rtol=tol["rel"]
    )


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("wrho", "vel", 1.69),
        ("wrho", "vel_a", (1.69, 2.1025)),
        ("wrho_a", "vel", (1.69, 1.859)),
        ("wrho_a", "vel_a", (1.69, 2.31275)),
    ],
    indirect=True,
)
def test_wat_bulkmod(dummy_values, tol):
    param, ans = dummy_values
    rho, vel = param
    assert np.allclose(bw92.wat_bulkmod(rho, vel), ans, rtol=tol["rel"])


# def test_mixed_density(den, frac, *argv):
#     param, ans = dummy_values
#     p, t, G = param


@pytest.mark.parametrize(
    "dummy_values",
    [
        ("wrho", "vel", 1.69),
        ("wrho", "vel_a", (1.69, 2.1025)),
        ("wrho_a", "vel", (1.69, 1.859)),
        ("wrho_a", "vel_a", (1.69, 2.31275)),
    ],
    indirect=True,
)
def test_bulkmod(dummy_values, tol):
    param, ans = dummy_values
    rho, vel = param
    assert np.allclose(bw92.bulkmod(rho, vel), ans, rtol=tol["rel"])


# def test_mixed_bulkmod(mod, frac, *argv):
#     param, ans = dummy_values
#     p, t, G = param
