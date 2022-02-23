import pytest

import numpy as np

from digirock import WaterECL, WaterBW92


@pytest.fixture(scope="session")
def mock_WaterBW92():
    return WaterBW92(name="mock", salinity=1e5)


def test_Water_get_summary(mock_WaterBW92):
    assert isinstance(mock_WaterBW92.get_summary(), dict)


def test_Water_density(mock_WaterBW92, test_props, tol):
    ans = np.full(
        test_props["shp"],
        1.0490675,
    )
    assert np.allclose(
        mock_WaterBW92.density(test_props["props"]), ans, rtol=tol["rel"]
    )


def test_Water_velocity(mock_WaterBW92, test_props, tol):
    ans = np.full(
        test_props["shp"],
        1711.353488089145,
    )
    assert np.allclose(
        mock_WaterBW92.velocity(test_props["props"]), ans, rtol=tol["rel"]
    )


def test_Water_bulk_modulus(mock_WaterBW92, test_props, tol):
    ans = np.full(
        test_props["shp"],
        3.07243626,
    )
    assert np.allclose(
        mock_WaterBW92.bulk_modulus(test_props["props"]), ans, rtol=tol["rel"]
    )


@pytest.fixture(scope="session")
def mock_WaterECL():
    return WaterECL(26.85, 1.03382, 0.31289e-04, 0.38509, 0.97801e-04)


def test_WaterECL_properties(mock_WaterECL):
    assert mock_WaterECL.density_asc == 0.9980103952594717


def test_WaterECL_get_summary(mock_WaterECL):
    assert isinstance(mock_WaterECL.get_summary(), dict)


def test_WaterECL_density(mock_WaterECL, test_props, tol):
    ans = np.full(
        test_props["shp"],
        1.0490675,
    )
    assert np.allclose(mock_WaterECL.density(test_props["props"]), ans, rtol=tol["rel"])


def test_WaterECL_velocity(mock_WaterECL, test_props, tol):
    ans = np.full(
        test_props["shp"],
        1711.353488089145,
    )
    assert np.allclose(
        mock_WaterECL.velocity(test_props["props"]), ans, rtol=tol["rel"]
    )


def test_WaterECL_bulk_modulus(mock_WaterECL, test_props, tol):
    ans = np.full(
        test_props["shp"],
        2.6818897153432397,
    )
    assert np.allclose(
        mock_WaterECL.bulk_modulus(test_props["props"]), ans, rtol=tol["rel"]
    )
