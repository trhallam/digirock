import pytest
from hypothesis import given
from hypothesis.strategies import composite, floats, integers, one_of
from hypothesis.extra.numpy import arrays, array_shapes, broadcastable_shapes

import numpy as np

from digirock.elastic import acoustic_moduli, acoustic_vel, poisson_ratio


@composite
def np_ints_or_floats(draw, shp=None, min_value=1.0, max_value=1.0e10):
    if shp is None:
        shp = (1,)
    kwargs = dict(min_value=min_value, max_value=max_value)
    # i = draw(integers(min_value=0, max_value=3))
    return draw(
        one_of(
            arrays(
                float,
                shape=shp,
                elements=floats(**kwargs),
            ),
            arrays(
                int,
                shape=shp,
                elements=integers(min_value=min_value, max_value=max_value),
            ),
            floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=min_value,
                max_value=max_value,
            ),
            integers(min_value=min_value, max_value=max_value),
        )
    )


@composite
def n_varshp_arrays(draw, n):
    # create a base shape
    shp = draw(array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5))
    # create a set of broadcast compatable shapes
    shps = broadcastable_shapes(shp, min_dims=1)
    shps = tuple(draw(shps) for _ in range(n))
    return tuple(draw(np_ints_or_floats(shp=shp)) for shp in shps)


@pytest.mark.parametrize(
    "vp,vs,rhob,k_ans,mu_ans", ((2, 1, 2, 16 / 3 * 1e-6, 2 * 1e-6),)
)
def test_acoustic_moduli(vp, vs, rhob, k_ans, mu_ans):
    k, mu = acoustic_moduli(vp, vs, rhob)
    assert np.allclose(k, k_ans)
    assert np.allclose(mu, mu_ans)

    for shp in ((2,), (2, 3), (2, 3, 4)):
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
def test_acoustic_vel(vp_ans, vs_ans, rhob, k, mu):
    vp, vs = acoustic_vel(k, mu, rhob)
    assert np.allclose(vp, vp_ans)
    assert np.allclose(vs, vs_ans)

    for shp in ((2,), (2, 3), (2, 3, 4)):
        vp, vs = acoustic_vel(
            np.full(shp, k),
            np.full(shp, mu),
            np.full(shp, rhob),
        )
        assert np.allclose(np.full(shp, vp_ans), vp)
        assert np.allclose(np.full(shp, vs_ans), vs)


# TODO: broadcatable from hypothesis seems tempremental
# @given(n_varshp_arrays(3))
# def test_vel_modulus_inverse(s):
#     s = tuple(map(lambda x: np.abs(x) + 0.001, s))
#     # velocity first because always creates positive values
#     vp, vs = acoustic_vel(*s)
#     k, mu = acoustic_moduli(vp, vs, s[2])
#     assert np.allclose(k, s[0], atol=1e-5)
#     assert np.allclose(mu, s[1], atol=1e-5)
