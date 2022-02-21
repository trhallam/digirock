from telnetlib import EL
from unittest import mock
import pytest
import numpy as np


from digirock._base import (
    Element,
    _element_check,
    _volume_sum_check,
    _get_complement,
    Switch,
    Blend,
)


@pytest.fixture(scope="module", params=[{"name": None}, {"name": "test"}])
def mock_Element(request):
    name = request.param["name"]
    words = "hello world"
    test = Element(name)
    for word in words.split(" "):
        test.register_key(word)

    assert test.name == name
    return test


def test_Element_keys(mock_Element):
    assert mock_Element.keys() == ["hello", "world"]


def test_Element_duplicate_register(mock_Element):
    with pytest.raises(ValueError):
        mock_Element.register_key("hello")


def test_Element_not_registered(mock_Element):
    with pytest.raises(ValueError):
        mock_Element.deregister_key("asdfds")


def test_Element_deregister(mock_Element):
    mock_Element.deregister_key("hello")
    assert mock_Element.keys() == ["world"]


def test_Element_tree(mock_Element):
    isinstance(str(mock_Element.tree), str)


def test_element_check():
    e1 = Element("test1")
    e1.attr1 = 1
    e1.attr2 = 2
    e2 = Element("test2")
    e2.attr1 = 3
    e2.attr2 = [4, 3]
    _element_check([e1], ["attr1", "attr2"])
    _element_check([e1, e2], ["attr1", "attr2"])
    assert True


def test_element_check_fail():
    e1 = Element("test1")
    e1.attr1 = 1
    e1.attr2 = 2
    e2 = Element("test2")
    e2.attr1 = 3
    with pytest.raises(ValueError):
        _element_check([e1, e2], ["attr1", "attr2"])


@pytest.mark.parametrize(
    "props",
    (
        {"a": 1, "b": 0},
        {"a": np.r_[1, 1], "b": 0},
        {"a": np.r_[1, 1], "b": np.r_[0, 0]},
        {"a": np.r_[0.5, 0.5], "b": np.r_[0.5, 0.5]},
        {"a": np.tile(np.r_[0.5, 0.5], (2, 2)), "b": np.tile(np.r_[0.5, 0.5], (2, 2))},
    ),
)
def test_volume_sum_check(props):
    _volume_sum_check(props)
    assert True


@pytest.mark.parametrize(
    "props",
    (
        {"a": 1, "b": 0.1},
        {"a": np.r_[1, 1], "b": 0.1},
        {"a": np.r_[1, 1], "b": np.r_[0.1, 0]},
        {"a": np.r_[0.5, 0.5], "b": np.r_[0.6, 0.5]},
        {"a": np.tile(np.r_[0.3, 0.5], (2, 2)), "b": np.tile(np.r_[0.7, 0.6], (2, 2))},
    ),
)
def test_volume_sum_check_fail(props):
    with pytest.raises(ValueError):
        _volume_sum_check(props)


# fmt: off
@pytest.mark.parametrize(
    "props,comp",
    (
        ({"a": 1, "b": 0,}, 0),
        ({"a": 0, "b": 0,}, 1),
        ({"a": np.r_[1, 1], "b": 0}, np.r_[0, 0]),
        ({"a": np.r_[0, 0], "b": 0}, np.r_[1, 1]),
        ({"a": np.r_[1, 1], "b": np.r_[0, 0]}, np.r_[0, 0]),
        ({"a": np.r_[0.5, 0.5], "b": np.r_[0.2, 0.2]}, np.r_[0.3, 0.3]),
        ({"a": np.r_[1.5, 1.5], "b": np.r_[0.2, 0.2]}, np.r_[0.0, 0.0]),
        (
            {
                "a": np.tile(np.r_[0.2, 0.5], (2, 2)),
                "b": np.tile(np.r_[0.5, 0.2], (2, 2)),
            },
            np.tile(np.r_[0.3, 0.3], (2, 2)),
        ),
    ),
)
# fmt: on
def test_get_complement(props, comp):
    assert np.allclose(comp, _get_complement(props))


@pytest.fixture(scope="module")
def mock_Switch():
    e1 = Element("test1", keys=["e1_key"])
    e1.attr1 = lambda props, **kwargs: 11.0
    e1.attr2 = lambda props, **kwargs: 12.0
    e2 = Element("test2", keys=["e2_key"])
    e2.attr1 = lambda props, **kwargs: 21.0
    e2.attr2 = lambda props, **kwargs: 22.0

    switch = Switch("test_key", [e1, e2], methods=["attr1", "attr2"])
    return switch


def test_Switch_properties(mock_Switch):
    assert mock_Switch.n_elements == 2
    for el in mock_Switch.elements:
        assert isinstance(el, Element)
    assert mock_Switch.methods == ["attr1", "attr2"]
    assert mock_Switch.switch_key == "test_key"


def test_Switch_reporting(mock_Switch):
    assert isinstance(mock_Switch.get_summary(), dict)
    mock_Switch.tree
    for key in ["test_key", "e1_key", "e2_key"]:
        assert key in mock_Switch.all_keys()
    assert True


def test_Switch_methods(mock_Switch):
    assert np.allclose(mock_Switch.attr1({"test_key": [0, 1]}), np.array([11, 21]))
    assert np.allclose(mock_Switch.attr2({"test_key": [1, 0]}), np.array([22, 12]))

    with pytest.raises(ValueError):
        mock_Switch.attr2({"wrong_test_key": [1, 0]})

    with pytest.raises(ValueError):
        mock_Switch.attr2({"test_key": [2, 0]})


@pytest.fixture(scope="module", params=[{"name": None}, {"name": "test"}])
def mock_Blend(request):
    e3 = Element("test1", keys=["e3_key"])
    e3.attr1 = lambda props, **kwargs: 31.0
    e3.attr2 = lambda props, **kwargs: 32.0
    e4 = Element()
    e4.attr1 = lambda props, **kwargs: 41.0
    e4.attr2 = lambda props, **kwargs: 42.0

    name = request.param["name"]
    blend = Blend(
        ["test_b_key1", "test_b_key2"],
        [e3, e4],
        methods=["attr1", "attr2"],
        name=name,
    )
    assert blend.name == name
    return blend


def test_Blend_properties(mock_Blend):
    assert mock_Blend.n_elements == 2
    for el in mock_Blend.elements:
        assert isinstance(el, Element)
    assert mock_Blend.methods == ["attr1", "attr2"]
    assert mock_Blend.blend_keys == ["test_b_key1", "test_b_key2"]


def test_Blend_reporting(mock_Blend):
    assert isinstance(mock_Blend.get_summary(), dict)
    mock_Blend.tree
    for key in ["test_b_key1", "test_b_key2", "e3_key"]:
        assert key in mock_Blend.all_keys()
    assert True
