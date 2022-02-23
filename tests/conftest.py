import pathlib
import pytest

import numpy as np


@pytest.fixture(scope="package")
def test_data():
    return pathlib.Path(__file__).parent / "test_data"


@pytest.fixture(
    params=[
        {"id": 0, "props": {"temp": 100, "pres": 50}},
        {"id": 1, "props": {"temp": np.r_[100, 100], "pres": 50}},
        {"id": 2, "props": {"temp": np.r_[100, 100], "pres": np.r_[50, 50]}},
        {"id": 3, "props": {"temp": 100, "pres": np.r_[50, 50]}},
    ],
    scope="package",
)
def test_props(request):
    shps = [np.atleast_1d(p).shape for p in request.param["props"].values()]
    final_shp = np.broadcast_shapes(*shps)
    request.param["shp"] = final_shp
    return request.param


@pytest.fixture
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }
