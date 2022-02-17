import pytest

from digirock.utils.ecl import (
    InitIntheadMap,
    EclStandardConditions,
    EclUnitMap,
    EclUnitScaler,
)

# mainly testing initialisation
def test_InitIntheadMap():
    assert InitIntheadMap["UNITS"].value == 2


@pytest.mark.parametrize("unit", ("METRIC", "FIELD", "LAB", "PVTM"))
def test_EclUnitMap(unit):
    assert unit in EclUnitMap._member_names_


@pytest.mark.parametrize("k,v", (("TEMP", 15.5556), ("PRES", 0.101325)))
def test_EclStandardConditions(k, v):
    assert EclStandardConditions[k].value == v


def test_EclUnitScalar():
    for k in EclUnitMap._member_names_:
        assert k in EclUnitScaler._member_names_
        assert isinstance(EclUnitScaler[k].value, dict)
