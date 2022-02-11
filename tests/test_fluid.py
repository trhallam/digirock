"""Test functions for digirock._fluid module
"""
import pytest
from pytest import approx
from _pytest.fixtures import SubRequest

import numpy as np
import xarray as xr
import digirock._fluid as fl
from digirock import Fluid, Water, WaterECL, DeadOil, Oil, FluidModel
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


@pytest.mark.parametrize("f", ["density", "velocity", "modulus"])
def test_Fluid_prototypes(f):
    with pytest.raises(PrototypeError):
        Fluid().__getattribute__(f)(None, None)


def test_Water_init():
    assert isinstance(Water(), Water)


@pytest.mark.parametrize(
    "f, ans", [("density", 1.01422375), ("velocity", 5397.0), ("modulus", 29.541912)]
)
@pytest.mark.parametrize("n", [1, 10, (10, 10)])
@pytest.mark.parametrize("temp, pres", [(150, 300)])
def test_Water_elastics(f, n, temp, pres, ans, tol):
    temp = np.full(n, temp)
    pres = np.full(n, pres)
    assert np.allclose(Water().__getattribute__(f)(temp, pres), ans, rtol=tol["rel"])


# def test_WaterECL_init():
#     assert isinstance(WaterECL("test", 100_000), WaterECL)


# def test_WaterECL_set_active_pvtn_error1():
#     test = WaterECL("test", 100_000)
#     with pytest.raises(WorkflowError):
#         test.set_active_pvtn(1)


# @pytest.fixture(scope="module")
# def mock_WaterECL(test_data):
#     test = WaterECL("test", 100_000)
#     test.load_pvtw(test_data / "COMPLEX_PVT.inc")
#     return test


# def test_WaterECL_set_active_pvtn_error2(mock_WaterECL):
#     with pytest.raises(ValueError):
#         mock_WaterECL.set_active_pvtn(-1)
#     with pytest.raises(ValueError):
#         mock_WaterECL.set_active_pvtn(100)


# @pytest.mark.parametrize(
#     "f, ans", [("density", 1.10991312), ("velocity", 1660.47), ("modulus", 3.0602174)]
# )
# @pytest.mark.parametrize("n", [1, 10, (10, 10)])
# @pytest.mark.parametrize("temp, pres, pvtn", [(100, 25, 0)])
# def test_WaterECL_elastics(mock_WaterECL, f, n, temp, pres, pvtn, ans, tol):
#     mock_WaterECL.set_active_pvtn(0)
#     temp = np.full(n, temp)
#     pres = np.full(n, pres)
#     pvtn = np.full(n, pvtn)
#     assert np.allclose(
#         mock_WaterECL.__getattribute__(f)(temp, pres), ans, rtol=tol["rel"]
#     )
#     assert np.allclose(
#         mock_WaterECL.__getattribute__(f)(temp, pres, pvt=pvtn), ans, rtol=tol["rel"]
#     )


@pytest.fixture()
def mock_deadoil(request: SubRequest, test_data):
    param = getattr(request, "param", None)
    test = DeadOil(api=35)

    pvt = param[:-1][0]
    ans = param[-1]

    if pvt == "const":
        test.set_fvf(1.4)
    elif pvt == "calc":
        test.calc_fvf(120)
    elif pvt == "text":
        pvt = np.loadtxt(test_data / "PVT_BO.inc")
        test.set_fvf(pvt[:, 1], pvt[:, 0])
    else:
        raise ValueError(f"Unknown pvt key {pvt} in fixture")
    yield test, ans


class TestDeadOil:
    def test_init(self):
        assert isinstance(DeadOil(), DeadOil)

    @pytest.mark.parametrize(
        "api, rho, ans",
        [
            (None, None, (None, None)),
            (30, None, (30, 0.8761609)),
            (None, 0.6, (104.333333, 0.6)),
        ],
    )
    def test_init_kw(self, api, rho, ans):
        test = Oil(api=api, std_density=rho)
        try:
            assert (test.api, test.std_density) == approx(ans)
        except TypeError:
            assert (test.api, test.std_density) == ans

    @pytest.mark.parametrize(
        "mock_deadoil",
        [
            ("const", float),
            ("calc", float),
            ("text", xr.DataArray),
        ],
        indirect=True,
    )
    def test_set_fvf(self, mock_deadoil):
        MockDeadOil, ans = mock_deadoil
        assert isinstance(MockDeadOil.bo, ans)

    @pytest.mark.parametrize(
        "mock_deadoil",
        [
            ("const", float),
            ("calc", float),
        ],
        indirect=True,
    )
    def test_fvf(self, mock_deadoil):
        MockDeadOil, ans = mock_deadoil
        assert isinstance(MockDeadOil.fvf(), ans)

    @pytest.mark.parametrize(
        "pres, fvf",
        [
            (0, 1.08810526),
            (100, 1.248),
            (200, 1.398),
            (300, 1.467),
            (500, 1.427),
            (np.r_[100, 200], np.r_[1.248, 1.398]),
        ],
    )
    @pytest.mark.parametrize("mock_deadoil", [("text", None)], indirect=True)
    def test_fvf_table(self, mock_deadoil, pres, fvf):
        MockDeadOil, ans = mock_deadoil
        assert MockDeadOil.fvf(pres) == approx(fvf)

    @pytest.mark.parametrize("mock_deadoil", [("calc", 1.095994)], indirect=True)
    def test_calc_fvf(self, mock_deadoil):
        MockDeadOil, ans = mock_deadoil
        assert MockDeadOil.fvf() == approx(ans)

    # def test_density(self, temp, pres):

    # def test_velocity(self, temp, pres):

    # def test_modulus(self, temp, pres):


@pytest.fixture()
def mock_oil(request: SubRequest, test_data):
    param = getattr(request, "param", None)
    test = Oil(api=35)

    pvt = param[:-1][0]
    ans = param[-1]

    if pvt == "const":
        test.set_disolved_gas(0.9, 100)
        test.set_fvf(1.4, 100)
    elif pvt == "calc":
        test.set_disolved_gas(0.9, 100)
        test.calc_fvf(120, 300)
    elif pvt == "text":
        pvt = np.loadtxt(test_data / "PVT_BO.inc")
        test.set_disolved_gas(0.9, 100)
        test.set_fvf(pvt[:, 1], 100, pvt[:, 0])
    elif pvt == "text2":
        pvt = np.loadtxt(test_data / "PVT_BO.inc")
        test.set_disolved_gas(0.9, 100)
        test.set_fvf(pvt[:, 1], 100, pvt[:, 0])
        test.set_fvf(pvt[:, 1], 120, pvt[:, 0])
    elif pvt == "pvto":
        test.set_disolved_gas(0.9, 100)
        test.load_pvto(test_data / "COMPLEX_PVT.inc")
    else:
        raise ValueError(f"Unknown pvt key {pvt} in fixture")
    yield test, ans


class TestOil:
    def test_init(self):
        assert isinstance(Oil(), Oil)

    @pytest.mark.parametrize(
        "api, rho, ans",
        [
            (None, None, (None, None)),
            (30, None, (30, 0.8761609)),
            (None, 0.6, (104.333333, 0.6)),
        ],
    )
    def test_init_kw(self, api, rho, ans):
        test = Oil(api=api, std_density=rho)
        try:
            assert (test.api, test.std_density) == approx(ans)
        except TypeError:
            assert (test.api, test.std_density) == ans

    @pytest.mark.parametrize(
        "mock_oil", [("text", xr.DataArray), ("text2", xr.DataArray)], indirect=True
    )
    def test_set_fvf(self, mock_oil):
        mockOil, ans = mock_oil
        assert isinstance(mockOil.bo, ans)

    @pytest.mark.parametrize("mock_oil", [("pvto", xr.DataArray)], indirect=True)
    def test_load_pvto(self, mock_oil):
        mockOil, ans = mock_oil
        assert isinstance(mockOil.bo, ans)

    # @pytest.mark.parametrize(
    #     "mock_oil", [("const", float), ("calc", float)], indirect=True
    # )
    # def test_calc_fvf_type(self, mock_oil):
    #     mockOil, ans = mock_oil
    #     assert isinstance(mockOil.fvf(100), ans)

    # @pytest.mark.parametrize(
    #     "mock_oil",
    #     [("const", 1.4), ("calc", 1.386388), ("text", 1.467), ("pvto", 1.34504596)],
    #     indirect=True,
    # )
    # def test_calc_fvf(self, mock_oil):
    #     mockOil, ans = mock_oil
    #     assert mockOil.fvf(300) == approx(ans)

    # def test_
