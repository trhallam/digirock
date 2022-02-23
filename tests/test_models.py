"""Test functions for pem.mod module

These test functions are designed to test core functionality with pytest

"""

# pylint: disable=invalid-name,redefined-outer-name,missing-docstring

import pytest
# from pytest import approx
from _pytest.fixtures import SubRequest

import numpy as np
import digirock.models as mod
# from inspect import getmembers, isfunction

@pytest.fixture
def tol():
    return {
        'rel': 0.05,   # relative testing tolerance in percent
        'abs': 0.00001 # absolute testing tolerance
    }

@pytest.fixture
def dummy_values(request: SubRequest):
    param = getattr(request, 'param', None)
    keys = param[:-1]
    ans = param[-1]
    array_size = 10
    values = {
        'm'   : 20,
        'f'   : 0.2,
        'k1'  : 37,
        'k2'  : 25,
        'mu1' : 22,
        'mu2' : 15,
        'phi' : 0.15,
        'vs'  : 2500,
        'vp'  : 3500,
        'rho' : 2.8,
        'kdry': 25,
        'ksat': 18,
        'kfl' : 20,
        'k0'  : 37,
        'erp_init': 12,
        'erp' : 15,
        'mod_vrh' : 32,
        'mod_e' : 12,
        'mod_p' : 10
    }
    arrays = {key+'a':np.full(array_size, val) for key, val in values.items()}
    values.update(arrays)
    if keys:
        v = tuple(values[k] for k in keys)
    else:
        v = tuple(values.values)
    yield v, ans

@pytest.mark.parametrize('dummy_values', [
    ('m', 'fa', (27, 27))
], indirect=True)
def test_voigt_upper_bound(dummy_values, tol):
    param, ans = dummy_values
    m, f = param
    ans = np.full_like(f, m)
    f2 = 1 - f
    f = np.vstack([f, f2]).T
    m = np.full(2, m)
    assert np.allclose(
        mod.voigt_upper_bound(m, f),
        ans, rtol=tol['rel']
    )

@pytest.mark.parametrize('dummy_values', [
    ('m', 'fa', (27, 27))
], indirect=True)
def test_reuss_lower_bound(dummy_values, tol):
    param, ans = dummy_values
    m, f = param
    ans = np.full_like(f, m)
    f2 = 1 - f
    f = np.vstack([f, f2]).T
    m = np.full(2, m)
    assert np.allclose(
        mod.reuss_lower_bound(m, f),
        ans, rtol=tol['rel']
    )

@pytest.mark.parametrize('dummy_values', [
    ('k1', 'k2', 'mu1', 'f', 27.03962),
    ('k1a', 'k2a', 'mu1a', 'f', 27.03962)
], indirect=True)
def test_hs_kbounds(dummy_values, tol):
    param, ans = dummy_values
    k1, k2, mu1, f = param
    ans = np.full_like(k1, ans)
    assert np.allclose(
        mod.hs_kbounds(k1, k2, mu1, f),
        ans, rtol=tol['rel']
    )

@pytest.mark.parametrize('dummy_values', [
    ('k1', 'k2', 'mu1', 'f', 22.57115779),
    ('k1a', 'k2a', 'mu1a', 'f', 22.57115779)
], indirect=True)
def test_hs_mubound(dummy_values, tol):
    param, ans = dummy_values
    k1, mu1, mu2, f = param
    ans = np.full_like(k1, ans)
    assert np.allclose(
        mod.hs_mubounds(k1, mu1, mu2, f),
        ans, rtol=tol['rel']
    )

@pytest.mark.parametrize('dummy_values', [
    ('k1', 'k2', 'f', 27.067052),
    ('k1', 'k2', 'fa', 27.067052),
    ('k1', 'k2', 'f', 27.067052),
    ('k1', 'k2', 'fa', 27.067052)
], indirect=True)
def test_vrh_avg(dummy_values, tol):
    param, ans = dummy_values
    k1, k2, f = param
    n1 = np.array(f).size
    ans = np.full(n1, ans)
    assert np.allclose(
        mod.vrh_avg(k1, k2, f),
        ans, rtol=tol['rel']
    )

# def test_acoustic_vel():
#     assert mod.acoustic_vel(test_vars['k1'],
#                             test_vars['mu1'],
#                             test_vars['rho']) == approx((result_vars['vp'], result_vars['vs']))
#     vp, vs = mod.acoustic_vel(test_vars_array['k1'], test_vars_array['mu1'], test_vars_array['rho'])
#     assert vst == approx(vs)
#     assert vpt == approx(vp)

# def test_acoustic_moduli():
#     assert mod.acoustic_moduli(vp, vs, rho) == approx((10.575000000000001, 16.875))

# def test_dryframe_delta_pres():
#     assert mod.dryframe_delta_pres(erp_init, erp, mod_vrh, mod_e, mod_p, phi, c=[40,0,40,0,15]) == approx(40.0202865)
#     phs = np.array([phi, phi])
#     assert mod.dryframe_delta_pres(erp_init, erp, mod_vrh, mod_e, mod_p, phs) == approx(np.array([ 40.0202865,  40.0202865]))

# def test_dryframe_acoustic():
#     assert mod.dryframe_acoustic(ksat, kfl, k0, phi) == approx(43.275687)
#     ksats = np.array([ksat, ksat])
#     fls   = np.array([kfl, kfl])
#     kms   = np.array([k0, k0])
#     pors  = np.array([phi, phi])
#     assert mod.dryframe_acoustic(ksats, fls, kms, pors) == approx(np.array([43.275687, 43.275687]))

# def test_gassmann_fluidsub():
#     assert mod.gassmann_fluidsub(kdry, kfl, k0, phi) == approx(33.613728)
#     kds = np.array([kdry, kdry])
#     kfs = np.array([kfl, kfl])
#     k0s = np.array([k0, k0])
#     phs = np.array([phi, phi])
#     assert mod.gassmann_fluidsub(kds, kfs, k0s, phs) == approx(np.array([33.613728, 33.613728]))

# def test_func_has_test():
#     function_test_record_ver = {func:True for func in function_test_record.keys()}
#     assert function_test_record == function_test_record_ver

if __name__ == "__main__":
    """Perform local debugging or testing during development here.
    """
    pass
