import pytest
from digirock import fluids

from digirock._base import BaseModelClass


@pytest.fixture(scope="module")
def mock_BaseModelClass():
    words = "hello world"
    test = BaseModelClass("test")
    for word in words.split(" "):
        test.register_key(word)
    return test


def test_BaseModelClass_name(mock_BaseModelClass):
    assert mock_BaseModelClass.name == "test"


def test_BaseModelClass_keys(mock_BaseModelClass):
    assert mock_BaseModelClass.keys() == ["hello", "world"]


def test_BaseModelClass_duplicate_register(mock_BaseModelClass):
    with pytest.raises(ValueError):
        mock_BaseModelClass.register_key("hello")


def test_BaseModelClass_not_registered(mock_BaseModelClass):
    with pytest.raises(ValueError):
        mock_BaseModelClass.deregister_key("asdfds")


def test_BaseModelClass_deregister(mock_BaseModelClass):
    mock_BaseModelClass.deregister_key("hello")
    assert mock_BaseModelClass.keys() == ["world"]
