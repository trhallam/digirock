from hypothesis import strategies as st
from hypothesis.extra import numpy as stn

import numpy as np


@st.composite
def np_ints_or_floats(draw, shp=None, min_value=1.0, max_value=1.0e10):
    """Strategy for generating NDArrayOrFloat types."""
    if shp is None:
        shp = (1,)
    kwargs = dict(min_value=min_value, max_value=max_value)
    # i = draw(integers(min_value=0, max_value=3))
    return draw(
        st.one_of(
            stn.arrays(
                np.float64,
                shape=shp,
                elements=st.floats(**kwargs),
            ),
            stn.arrays(
                np.int64,
                shape=shp,
                elements=st.integers(**kwargs),
            ),
            st.floats(allow_nan=False, allow_infinity=False, **kwargs),
            st.integers(**kwargs),
        )
    )


@st.composite
def n_varshp_arrays(draw, n, min_value=-1e10, max_value=1e10):
    """Strategy for generate broadcastable sets of arrays."""

    if isinstance(n, tuple):
        num_shapes = len(n)
    else:
        num_shapes = n

    # create a base shape
    shp = draw(stn.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5))
    # create a set of broadcast compatable shapes
    shps = draw(
        stn.mutually_broadcastable_shapes(
            num_shapes=num_shapes - 1, base_shape=shp, min_dims=1
        )
    )
    input_shapes = [shp] + list(shps.input_shapes)
    result_shape = shps.result_shape
    print(input_shapes)
    if isinstance(n, tuple):
        return tuple(np.full(bc_shp, val) for bc_shp, val in zip(input_shapes, n))
    else:
        return (
            tuple(
                draw(
                    np_ints_or_floats(
                        shp=bc_shp, min_value=min_value, max_value=max_value
                    )
                )
                for bc_shp in input_shapes
            ),
            result_shape,
        )
