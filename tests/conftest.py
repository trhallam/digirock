import pathlib
import pytest

@pytest.fixture(scope="package")
def test_data():
    return pathlib.Path(__file__).parent / "test_data"
