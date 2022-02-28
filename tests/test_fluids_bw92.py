"""Test functions for pem.fluid.bw92 module
"""
import pytest
from pytest import approx
from _pytest.fixtures import SubRequest
from hypothesis import given, settings, strategies as st

import numpy as np
import digirock.fluids.bw92 as bw92

from .strategies import n_varshp_arrays


@pytest.fixture(scope="module")
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


def test_GAS_R():
    assert bw92.GAS_R == 8.31441


# p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15), 0.00045422), ((50 * 1e6, 373.15), 0.00010747))
)
@given(data=st.data())
def test_gas_vmol(args, ans, data, tol):
    (test_p, test_t), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_vmol(test_t, test_p)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# p (MPa), t (degC), m (methane molecular weight)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 273.15, 16.04), 35313.5783218),
        ((50 * 1e6, 373.15, 16.04), 149248.08786351),
    ),
)
@given(data=st.data())
def test_gas_density(args, ans, data, tol):
    (test_p, test_t, test_m), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_density(test_m, test_t, test_p)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans",
    (((np.r_[10 * 1e6, 50 * 1e6], 273.15), 2e-08),),
)
def test_gas_isotherm_comp(args, ans, tol):
    v1, v2 = bw92.gas_vmol(args[1], args[0])
    assert bw92.gas_isotherm_comp(v1, v2, args[0][0], args[0][1]) == approx(ans)


# t (degC), m (methane molecular weight),
@pytest.mark.parametrize(
    "args,ans",
    (
        ((273.15, 16.04), 16.8278695),
        ((373.15, 16.04), 18.30335126),
    ),
)
@given(data=st.data())
def test_gas_isotherm_vp(args, ans, data, tol):
    (test_t, test_m), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_isotherm_vp(test_m, test_t)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# G spec grav
@pytest.mark.parametrize(
    "args,ans",
    (((0.56,), 4.665312),),  # methane
)
@given(data=st.data())
def test_gas_pseudocrit_pres(args, ans, data, tol):
    (test_G,), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_pseudocrit_pres(test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# fmt: off
# p (MPa), G spec grav
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 0.56,), 2143479.36429546,),
        ((50 * 1e6, 0.56), 10717396.82147732),
    ),
)
# fmt: on
@given(data=st.data())
def test_gas_pseudored_pres(args, ans, data, tol):
    (
        test_p,
        test_G,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_pseudored_pres(test_p, test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# G spec grav
@pytest.mark.parametrize(
    "args,ans",
    (((0.56,), 190.34),),  # methane
)
@given(data=st.data())
def test_gas_pseudocrit_temp(args, ans, data, tol):
    (test_G,), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_pseudocrit_temp(test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# t (degC), G (spec grav)
# fmt: off
@pytest.mark.parametrize(
    "args,ans",
    (
        ((273.15, 0.56,), 2.87012714,),
        ((373.15, 0.56), 3.39550278),
    ),
)
# fmt: on
@given(data=st.data())
def test_gas_pseudored_temp(args, ans, data, tol):
    (
        test_t,
        test_G,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_pseudored_temp(test_t, test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# p (MPa), t (degC), G (spec grav)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 273.15, 0.56), 0.5289487894),
        ((50 * 1e6, 373.15, 0.56), 0.46664469),
    ),
)
@given(data=st.data())
def test_gas_oga_density(args, ans, data, tol):
    (
        test_p,
        test_t,
        test_G,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_oga_density(test_t, test_p, test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


def test_gas_oga_density_warning():
    with pytest.warns(UserWarning):
        bw92.gas_oga_density(4.5, 4.5, 1)


# p (MPa), t (degC), G (spec grav)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 273.15, 0.56), 673174274.6197122),
        ((50 * 1e6, 373.15, 0.56), 1.87375111e10),
    ),
)
@given(data=st.data())
def test_gas_adiabatic_bulkmod(args, ans, data, tol):
    (
        test_p,
        test_t,
        test_G,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_adiabatic_bulkmod(test_t, test_p, test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# p (MPa), t (degC), G (spec grav)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 273.15, 0.56), 0.0204339351378),
        ((50 * 1e6, 373.15, 0.56), 0.03011878),
    ),
)
@given(data=st.data())
def test_gas_adiabatic_viscosity(args, ans, data, tol):
    (
        test_p,
        test_t,
        test_G,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.gas_adiabatic_viscosity(test_t, test_p / 1e6, test_G)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# p (MPa), rho (g/cc)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 0.8), 0.8068623025),
        ((10 * 1e6, 0.9), 0.90521056),
        ((50 * 1e6, 0.8), 0.83179781),
        ((50 * 1e6, 0.9),  0.92477031),
    ),
)
@given(data=st.data())
def test_oil_isothermal_density(args, ans, data, tol):
    (
        test_p,
        test_rho,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_isothermal_density(test_rho, test_p / 1e6)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# p (MPa), rho (g/cc), t (degC)
@pytest.mark.parametrize(
    "args,ans",
    (
        ((10 * 1e6, 0.8, 273.15), 0.63475419),
        ((10 * 1e6, 0.9, 273.15), 0.71212423),
        ((50 * 1e6, 0.8, 273.15),  0.65437082),
        ((50 * 1e6, 0.9, 273.15),   0.72751178),
        ((10 * 1e6, 0.8, 373.15), 0.57827437),
        ((10 * 1e6, 0.9, 373.15), 0.6487601),
        ((50 * 1e6, 0.8, 373.15), 0.59614553),
        ((50 * 1e6, 0.9, 373.15),  0.65437082),
    ),
)
@given(data=st.data())
def test_oil_density(args, ans, data, tol):
    (
        test_p,
        test_rho,
        test_t
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_density(test_rho, test_p / 1e6, test_t)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho (g/cc), G (spec_grav), rg (L/L), t (degC)
@pytest.mark.parametrize(
        "args,ans",
    (
        ((0.8, 0.56, 120, 273.15), 1.57823582),
    )
)
@given(data=st.data())
def test_oil_fvf(args, ans, data, tol):
    (
        test_rho,
        test_G,
        test_rg,
        test_t,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_fvf(test_rho, test_G, test_rg, test_t)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho (g/cc), G (spec_grav), p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans",
    (((0.8, 0.6, 50, 100), 415.709664),)
)
@given(data=st.data())
def test_oil_rg_rho(args, ans, data, tol):
    (
        test_rho,
        test_G,
        test_p,
        test_t,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_rg(test_rho, test_G, test_p, test_t, mode="rho")
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho (g/cc), G (spec_grav), p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans",
    (((45, 0.6, 50, 100), 415.709664),)
)
@given(data=st.data())
def test_oil_rg_api(args, ans, data, tol):
    (
        test_rho,
        test_G,
        test_p,
        test_t,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_rg(test_rho, test_G, test_p, test_t, mode="api")
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


def test_oil_rg_bad_mode():
    with pytest.raises(ValueError):
        assert bw92.oil_rg(1, 1, 1, 1, mode="bad_mode")


# rho0, g, rg, b0
@pytest.mark.parametrize(
    "args,ans",
    (
        ((0.8, 0.6, 50, 1.1), 0.76),
        ((0.9, 0.6, 70, 1.1), 0.864),
        ((0.9, 0.6, 70, 0.0), 0.0)
    ),
)
@given(data=st.data())
def test_oil_rho_sat(args, ans, data, tol):
    (
        test_rho0,
        test_g,
        test_rg,
        test_b0,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_rho_sat(test_rho0, test_g, test_rg, test_b0)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho0, rg, b0
@pytest.mark.parametrize(
    "args,ans",
    (
        ((0.8, 50, 1.1), 0.69264069264),
        ((0.9, 70, 1.1), 0.764655904),
        ((0.9, 70, 0.0), 0.0)),
)
@given(data=st.data())
def test_oil_rho_pseudo(args, ans, data, tol):
    (
        test_rho0,
        test_rg,
        test_b0,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_rho_pseudo(test_rho0, test_rg, test_b0)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# rho0,p,t,g,rg,b0
@pytest.mark.parametrize(
    "args,ans",
    (
        ((0.8, 50, 100, 0.6, 120), 1101.21832685),
    ),
)
@given(data=st.data())
def test_oil_velocity_nobo(args, ans, data, tol):
    (
        test_rho0,
        test_p,
        test_t,
        test_g,
        test_rg,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_velocity(test_rho0, test_p, test_t, test_g, test_rg, b0=None)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho0,p,t,g,rg,b0
@pytest.mark.parametrize(
    "args,ans",
    (
        ((0.8, 50, 100, 0.6, 120),  1206.74469093),
    ),
)
@given(data=st.data())
def test_oil_velocity_bo(args, ans, data, tol):
    (
        test_rho0,
        test_p,
        test_t,
        test_g,
        test_rg,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_velocity(test_rho0, test_p, test_t, test_g, test_rg, b0=np.r_[1.1])
    assert np.allclose(test, ans, rtol=tol["rel"])
    # assert np.squeeze(test).shape == result_shape


# rho, vp
@pytest.mark.parametrize(
    "args,ans",
    (
        ((0.8, 1200),  1.152),
    ),
)
@given(data=st.data())
def test_oil_bulkmod(args, ans, data, tol):
    (
        test_rho,
        test_vp,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.oil_bulkmod(test_rho, test_vp)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15), 1063.23709), ((50 * 1e6, 373.15), 847.72401465))
)
@given(data=st.data())
@settings(deadline=None) # due to njit
def test_wat_velocity_pure(args, ans, data, tol):
    (
        test_p,
        test_t,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.wat_velocity_pure(test_t, test_p / 1e6)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# p (MPa), t (degC), sal (ppm)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15, 32000), 1095.70072), ((50 * 1e6, 373.15, 150000), 980.48475247))
)
@given(data=st.data())
@settings(deadline=None) # due to njit
def test_wat_velocity_brine(args, ans, data, tol):
    (
        test_p,
        test_t,
        test_sal
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.wat_velocity_brine(test_t, test_p / 1e6, test_sal / 1e6)
    assert np.allclose(test, ans, rtol=tol["rel"])


# p (MPa), t (degC)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15), 0.77622433), ((50 * 1e6, 373.15), 0.66363597))
)
@given(data=st.data())
def test_wat_density_pure(args, ans, data, tol):
    (
        test_p,
        test_t,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.wat_density_pure(test_t, test_p / 1e6)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# p (MPa), t (degC), sal (ppm)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15, 32000), 0.80405636), ((50 * 1e6, 373.15, 150000), 0.79606398))
)
@given(data=st.data())
def test_wat_density_brine(args, ans, data, tol):
    (
        test_p,
        test_t,
        test_sal
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.wat_density_brine(test_t, test_p / 1e6, test_sal / 1e6)
    assert np.allclose(test, ans, rtol=tol["rel"])


# p (MPa), t (degC), sal (ppm)
@pytest.mark.parametrize(
    "args,ans", (((10 * 1e6, 273.15, 32000), None), ((50 * 1e6, 373.15, 150000), None))
)
def test_wat_salinity_brine(args, ans):
    (
        test_p,
        test_t,
        test_sal
    ) = args
    test_den = bw92.wat_density_brine(test_t, test_p / 1e6, test_sal / 1e6)
    test = bw92.wat_salinity_brine(test_t, test_p / 1e6, test_den) * 1e6
    assert test == approx(test_sal, abs=250)

# rho (g/cc), v (m/s)
@pytest.mark.parametrize(
    "args,ans", (((1.0, 1300), 1.69), ((1.1, 1450), 2.31275))
)
@given(data=st.data())
def test_wat_bulkmod(args, ans, data, tol):
    (
        test_rho,
        test_vp,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.wat_bulkmod(test_rho, test_vp)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# rho (g/cc), v (vfrac), ...
@pytest.mark.parametrize(
    "args,ans",
    (
        ((2, 0.5, 2, 0.5), 2),
        ((2, 0.5, 1, 0.5),  1.5),
        ((2, 0.3, 2, 0.3, 2,), 2),
        ((2, 0.5, 1, 0.5, 2,),  1.5),
    )
)
@given(data=st.data())
def test_mixed_density(args, ans, data, tol):
    args, result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.mixed_density(*args)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape


# rho (g/cc), v (m/s)
@pytest.mark.parametrize(
    "args,ans", (((1.0, 1300), 1.69), ((1.1, 1450), 2.31275))
)
@given(data=st.data())
def test_bulkmod(args, ans, data, tol):
    (
        test_rho,
        test_vp,
    ), result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.bulkmod(test_rho, test_vp)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape

# k (GPa), v (vfrac), ...
@pytest.mark.parametrize(
    "args,ans",
    (
        ((2, 0.5, 2, 0.5), 2),
        ((2, 0.5, 1, 0.5),  1.3333),
        ((2, 0.3, 2, 0.3, 2,), 2),
        ((2, 0.5, 1, 0.5, 2,),  1.3333),
    )
)
@given(data=st.data())
def test_woods_bulkmod(args, ans, data, tol):
    args, result_shape = data.draw(n_varshp_arrays(args))
    test = bw92.woods_bulkmod(*args)
    assert np.allclose(test, ans, rtol=tol["rel"])
    assert test.shape == result_shape
