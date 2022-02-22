import pytest
from hypothesis import given

from .strategies import n_varshp_arrays

import numpy as np

from digirock.elastic import acoustic_moduli, acoustic_vel, poisson_ratio


@pytest.fixture
def np_shapes():
    return ((2,), (2, 3), (2, 3, 4))


@pytest.mark.parametrize(
    "vp,vs,rhob,k_ans,mu_ans", ((2, 1, 2, 16 / 3 * 1e-6, 2 * 1e-6),)
)
def test_acoustic_moduli(vp, vs, rhob, k_ans, mu_ans, np_shapes):
    k, mu = acoustic_moduli(vp, vs, rhob)
    assert np.allclose(k, k_ans)
    assert np.allclose(mu, mu_ans)

    for shp in np_shapes:
        k, mu = acoustic_moduli(
            np.full(shp, vp),
            np.full(shp, vs),
            np.full(shp, rhob),
        )
        assert np.allclose(np.full(shp, k_ans), k)
        assert np.allclose(np.full(shp, mu_ans), mu)


@pytest.mark.parametrize(
    "vp_ans,vs_ans,rhob,k,mu", ((2, 1, 2, 16 / 3 * 1e-6, 2 * 1e-6),)
)
def test_acoustic_vel(vp_ans, vs_ans, rhob, k, mu, np_shapes):
    vp, vs = acoustic_vel(k, mu, rhob)
    assert np.allclose(vp, vp_ans)
    assert np.allclose(vs, vs_ans)

    for shp in np_shapes:
        vp, vs = acoustic_vel(
            np.full(shp, k),
            np.full(shp, mu),
            np.full(shp, rhob),
        )
        assert np.allclose(np.full(shp, vp_ans), vp)
        assert np.allclose(np.full(shp, vs_ans), vs)


@pytest.mark.parametrize("k,mu,pr", ((2, 1, 0.2857142857142857),))
def test_poisson_ratio(k, mu, pr, np_shapes):
    assert poisson_ratio(k, mu) == pr

    for shp in np_shapes:
        pr_ans = poisson_ratio(
            np.full(shp, k),
            np.full(shp, mu),
        )
        assert np.allclose(np.full(shp, pr), pr_ans)


# TODO: broadcatable from hypothesis seems tempremental
@given(n_varshp_arrays(3, min_value=1, max_value=50))
def test_vel_modulus_inverse(s):
    args, result_shape = s
    # s = tuple(map(lambda x: np.abs(x) + 0.001, s))
    # velocity first because always creates positive values
    vp, vs = acoustic_vel(*args)
    k, mu = acoustic_moduli(vp, vs, args[2])
    assert np.allclose(k, args[0], atol=1e-5)
    assert np.allclose(mu, args[1], atol=1e-5)
