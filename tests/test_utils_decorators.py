from tabnanny import check
import pytest

import numpy as np

from digirock.utils._decorators import (
    broadcastable,
    check_props,
    mutually_inclusive,
    mutually_exclusive,
)


@pytest.fixture(scope="module")
def test_array():
    l = 1
    l2 = np.r_[1, 1]
    l3 = np.r_[1, 1, 1]
    l22 = np.r_[l2, l2]
    return l, l2, l3, l22


def test_broadcastable(test_array):
    l, l2, l3, l22 = test_array

    @broadcastable("a", "b", "d")
    def tfunc(a, b, c, d=None, e=None):
        return True

    assert tfunc(l, l, l)
    assert tfunc(l2, l, l22, d=l, e=l3)

    with pytest.raises(ValueError):
        tfunc(l2, l22, l3, d=l3, e=1)


def test_check_props(test_array):
    l, l2, l3, l22 = test_array

    @check_props("t1", "t2")
    def tfunc(props, **kwargs):
        return True

    for p in (
        {"t1": l, "t2": l, "t3": l2},
        {"t1": l, "t2": l2, "t3": l2},
        {"t1": l22, "t2": l22, "t3": l22},
    ):
        assert tfunc(p)
