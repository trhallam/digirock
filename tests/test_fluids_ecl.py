"""Test functions for pem.fluid.ecl module
"""

import pytest
from pytest import approx
import numpy as np
import digirock.fluids.ecl as fluid_ecl
from inspect import getmembers, isfunction


@pytest.fixture
def tol():
    return {
        "rel": 0.05,  # relative testing tolerance in percent
        "abs": 0.00001,  # absolute testing tolerance
    }


@pytest.mark.parametrize(
    "pres, extrap, ans",
    [
        (325, "const", 1.4615),
        (325, "pchip", 1.4615),
        (np.r_[325, 375], "const", np.r_[1.4615, 1.4505]),
        (np.r_[325, 375], "pchip", np.r_[1.4615, 1.4505]),
    ],
)
def test_oil_fvf_table(test_data, pres, ans, extrap, tol):
    tab = np.loadtxt(test_data / "PVT_BO.inc")
    assert np.allclose(
        fluid_ecl.oil_fvf_table(tab[:, 0], tab[:, 1], pres, extrap=extrap),
        ans,
        rtol=tol["rel"],
    )


def test_oil_fvf_table_bad_pchi(test_data):
    tab = np.loadtxt(test_data / "PVT_BO.inc")
    # test bad extrap
    with pytest.raises(ValueError):
        assert fluid_ecl.oil_fvf_table(
            tab[:, 0], tab[:, 1], 235, extrap="Unknown Extrap"
        )


@pytest.mark.parametrize(
    "pres, extrap, ans",
    [
        (325, "const", 1.4615),
        (325, "pchip", 1.4615),
        (np.r_[325, 375], "const", np.r_[1.4615, 1.4505]),
        (np.r_[325, 375], "pchip", np.r_[1.4615, 1.4505]),
    ],
)
def test_oil_fvf_table(test_data, pres, ans, extrap, tol):
    tab = np.loadtxt(test_data / "PVT_BO.inc")
    assert np.allclose(
        fluid_ecl.oil_fvf_table(tab[:, 0], tab[:, 1], pres, extrap=extrap),
        ans,
        rtol=tol["rel"],
    )


@pytest.mark.parametrize("api,ans", ((20, 0.933993399339934), (45, 0.8016997167138812)))
def test_e100_oil_density(api, ans, tol):
    assert fluid_ecl.e100_oil_density(api) == approx(ans)
    assert np.allclose(
        fluid_ecl.e100_oil_density(np.r_[api, api]), np.r_[ans, ans], atol=tol["abs"]
    )
