import pytest

from digirock import load_pvdg, load_pvto, load_pvtw


@pytest.mark.parametrize("file,n", (("COMPLEX_PVT.inc", 12),))
def test_load_pvtw(test_data, file, n):
    pvtw = load_pvtw(test_data / file)
    assert len(pvtw) == n


@pytest.mark.parametrize("file,n", (("COMPLEX_PVT.inc", 12),))
def test_load_pvto(test_data, file, n):
    pvto = load_pvto(test_data / file, api=25)
    assert len(pvto) == n


@pytest.mark.parametrize("file,n", (("COMPLEX_PVT.inc", 12),))
def test_load_pvdg(test_data, file, n):
    pvdg = load_pvdg(test_data / file)
    assert len(pvdg) == n
