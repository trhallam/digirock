import pytest
import numpy as np


from digirock._base import (
    Element,
    _element_check,
    _volume_sum_check,
    _get_complement,
    Switch,
    Blend,
    Transform,
)


@pytest.fixture(scope="module", params=[{"name": None}, {"name": "test"}])
def mock_Element(request):
    name = request.param["name"]
    words = "hello world"
    test = Element(name)
    for word in words.split(" "):
        test.register_key(word)

    if name is None:
        assert isinstance(test.name, str)
    else:
        assert test.name == name

    test.test_function = lambda props, **kwargs: True

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
    assert isinstance(str(mock_Element.tree), str)


def test_Element_trace(mock_Element):
    trace = mock_Element.trace({}, "test_function")
    assert isinstance(trace, dict)
    assert trace["test_function"]


def test_Element_trace_tree(mock_Element):
    tree = mock_Element.trace_tree({}, "test_function")
    assert isinstance(str(tree), str)


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


def test_Switch_tree(mock_Switch):
    assert isinstance(str(mock_Switch.tree), str)


def test_Switch_trace(mock_Switch):
    _, trace1 = mock_Switch.trace({}, "attr1").popitem()
    assert isinstance(trace1, dict)
    assert trace1["test1"]["attr1"] == 11.0
    assert trace1["test2"]["attr1"] == 21.0

    _, trace2 = mock_Switch.trace({}, ["attr1", "attr2"]).popitem()
    assert trace2["test1"]["attr1"] == 11.0
    assert trace2["test2"]["attr1"] == 21.0
    assert trace2["test1"]["attr2"] == 12.0
    assert trace2["test2"]["attr2"] == 22.0


@pytest.fixture(scope="module", params=[{"name": None}, {"name": "test"}])
def mock_Blend(request):
    e3 = Element("test1", keys=["e3_key"])
    e3.attr1 = lambda props, **kwargs: 31.0
    e3.attr2 = lambda props, **kwargs: 32.0
    e4 = Element("test2")
    e4.attr1 = lambda props, **kwargs: 41.0
    e4.attr2 = lambda props, **kwargs: 42.0

    TestBlend = Blend
    setattr(
        TestBlend,
        "attr1",
        lambda self, props, **kwargs: sum(
            el.attr1(props, **kwargs) for el in self.elements
        ),
    )
    setattr(
        TestBlend,
        "attr2",
        lambda self, props, **kwargs: sum(
            el.attr2(props, **kwargs) for el in self.elements
        ),
    )

    name = request.param["name"]
    blend = TestBlend(
        ["test_b_key1", "test_b_key2"],
        [e3, e4],
        methods=["attr1", "attr2"],
        name=name,
    )
    if name is None:
        assert isinstance(blend.name, str)
    else:
        assert blend.name == name

    return blend


def test_Blend_properties(mock_Blend):
    assert mock_Blend.n_elements == 2
    for el in mock_Blend.elements:
        assert isinstance(el, Element)
    assert mock_Blend.methods == ("attr1", "attr2")
    assert mock_Blend.blend_keys == ["test_b_key1", "test_b_key2"]


def test_Blend_reporting(mock_Blend):
    assert isinstance(mock_Blend.get_summary(), dict)
    mock_Blend.tree
    for key in ["test_b_key1", "test_b_key2", "e3_key"]:
        assert key in mock_Blend.all_keys()
    assert True


def test_Blend_trace(mock_Blend):
    trace1 = mock_Blend.trace({}, "attr1")
    print(trace1)
    assert isinstance(trace1, dict)
    assert trace1["test1"]["attr1"] == 31.0
    assert trace1["test2"]["attr1"] == 41.0
    assert trace1["attr1"] == 72.0

    trace2 = mock_Blend.trace({}, ["attr1", "attr2"])
    assert trace2["test1"]["attr1"] == 31.0
    assert trace2["test2"]["attr1"] == 41.0
    assert trace2["test1"]["attr2"] == 32.0
    assert trace2["test2"]["attr2"] == 42.0
    assert trace2["attr1"] == 72.0
    assert trace2["attr2"] == 74.0


@pytest.fixture(scope="module", params=[{"name": None}, {"name": "test"}])
def mock_Transform(request):
    e5 = Element("test1", keys=["e3_key"])
    e5.attr1 = lambda props, **kwargs: 31.0
    e5.attr2 = lambda props, **kwargs: 32.0
    e5.attr3 = lambda props, **kwargs: 105.0

    name = request.param["name"]

    TestTsfm = Transform
    setattr(
        TestTsfm,
        "attr1",
        lambda self, props, **kwargs: self.element.attr1(props, **kwargs) + 1.0,
    )
    setattr(
        TestTsfm,
        "attr2",
        lambda self, props, **kwargs: self.element.attr2(props, **kwargs) + 2.0,
    )

    trsfm = TestTsfm(
        ["test_t_key1", "test_t_key2"],
        e5,
        methods=["attr1", "attr2", "attr3"],
        name=name,
    )
    if name is None:
        assert isinstance(trsfm.name, str)
    else:
        assert trsfm.name == name

    return trsfm


def test_Transform_properties(mock_Transform):
    assert isinstance(mock_Transform.element, Element)
    assert mock_Transform.elements == [mock_Transform.element]
    assert mock_Transform.methods == ("attr1", "attr2", "attr3")
    assert mock_Transform.transform_keys == ["test_t_key1", "test_t_key2"]


def test_Transform_reporting(mock_Transform):
    assert isinstance(mock_Transform.get_summary(), dict)
    mock_Transform.tree
    for key in ["test_t_key1", "test_t_key2", "e3_key"]:
        assert key in mock_Transform.all_keys()
    assert True


def test_Transform_trace(mock_Transform):
    trace1 = mock_Transform.trace({}, "attr1")
    assert isinstance(trace1, dict)
    assert trace1["attr1"] == 32.0
    assert trace1["test1"]["attr1"] == 31.0

    trace2 = mock_Transform.trace({}, ["attr1", "attr2"])
    assert trace2["test1"]["attr1"] == 31.0
    assert trace2["test1"]["attr2"] == 32.0
    assert trace2["attr1"] == 32.0
    assert trace2["attr2"] == 34.0


def test_Transform_getattr(mock_Transform):
    assert getattr(mock_Transform, "attr3")({}) == 105.0
    assert mock_Transform.attr3({}) == 105.0

    assert mock_Transform.attr2({}) == 34.0

    with pytest.raises(AttributeError):
        mock_Transform.attr4({})
