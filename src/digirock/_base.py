from typing import List, Dict, Union, Type, Any
from rich.tree import Tree

import numpy as np

from .utils import check_broadcastable
from .utils._decorators import check_props
from .utils.types import NDArrayOrFloat


class Element:
    """Base consumer class from which all other consumer classes are defined.
    For example, uses to define input consumer classes like fluids, mineral mixing models etc.



    Attributes:
       name (str): The name of the Model.
    """

    def __init__(self, name: str = None, keys: List[str] = None):
        self._protected_kw_registry = list()
        self.name = name
        if keys:
            for key in keys:
                try:
                    self.register_key(key)
                except ValueError:
                    pass  # ignore already registered keys if created using init

    def register_key(self, key: str):
        """Register a new keyword across digirock classes"""
        if key in self._protected_kw_registry:
            raise ValueError(
                f"The key: {key}, cannot be registered more than once. Use another key."
            )
        self._protected_kw_registry.append(key)

    def deregister_key(self, key: str):
        """Deregister a keyword across the digirock classes"""
        try:
            self._protected_kw_registry.remove(key)
        except ValueError:
            raise ValueError(
                f"The key: {key}, is not currently in the digirock registry."
            )

    def keys(self) -> List[str]:
        """Returns a list of keys this class will require for computation.

        Returns:
            The keys this class requires.
        """
        return self._protected_kw_registry

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        return {"class": self.__class__, "name": self.name, "props_keys": self.keys()}

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")
        if name is None:
            name = summary.pop("class")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        return tree


def _element_check(elements: List[Type[Element]], methods: List[str]):
    """Check elements have methods

    Raises:
        ValueError if any of elements don't have all of methods.
    """
    for i, el in enumerate(elements):
        for meth in methods:
            if getattr(el, meth, None) is None:
                raise ValueError(
                    f"The element `{el}` at position `{i}`, does not have required method `{meth}`"
                )


def _volume_sum_check(props: Dict[str, NDArrayOrFloat], sum_to=1, atol=1e-3) -> bool:
    """Check arrays all sum to no more than 1"""
    check_broadcastable(**props)
    sum_ar = np.zeros((1,))

    for prop in props:
        sum_ar = sum_ar + props[prop]

    try:
        assert sum_ar.max() <= sum_to + atol
    except AssertionError:
        raise ValueError(f"Volume fractions for {props.keys()} sum to greater than one")
    return True


def _get_complement(props: Dict[str, NDArrayOrFloat]) -> NDArrayOrFloat:
    """Find the ones complement to a bunch of arrays"""
    check_broadcastable(**props)
    sum_ar = np.zeros((1,))

    for prop in props:
        sum_ar = sum_ar + props[prop]

    return np.clip(1 - sum_ar, 0, 1)


class Switch(Element):
    """Base class for performing switching on other model classes e.g. `Element` or `Blend`.

    Switching works by passing a suitable NDArrayOrFloat in the `props` argument of any valid `Element`, the switch_key value
    should contain the index of desired element to use.

    Switches consume the `switch_key` prop from the props so it is not passed to it's children.

    Uses methods argument to build factory methods to map to Elements. Each Element must have a method matching the names in `methods`.

    Attributes:
        name (str): Name for switch
        switch_key (str): Key to use for switching
        elements (list): A list of elements
        methods (list): A list of methods that the Switch should implement to match the Elements.
        n_elements (int): The number of elements
    """

    def __init__(
        self,
        switch_key: str,
        elements: List[Type[Element]],
        methods: List[str],
        name: str = None,
    ):
        super().__init__(name=name, keys=[switch_key])
        self._methods = methods
        self._switch_key = switch_key
        self._elements = elements
        _element_check(self._elements, self.methods)

        for meth in self._methods:
            self._method_factory(meth)

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    @property
    def elements(self) -> List[Type[Element]]:
        return self._elements

    @property
    def methods(self) -> List[str]:
        return self._methods

    @property
    def switch_key(self) -> str:
        return self._switch_key

    def _method_factory(self, method_name: str):
        """Add required methods to switch."""

        docs = """
        Switch Factory Method for {0}
        
        Method requires props to contain `{0}`
            
        Args:
            props: A dictionary of properties to pass to elements of this switch.
            element_kwargs: passed to elements

        Returns:
            {0}
        """.format(
            method_name
        )

        @check_props(self.switch_key)
        def func(props: Dict[str, NDArrayOrFloat], **element_kwargs) -> NDArrayOrFloat:
            was_int = isinstance(props[self.switch_key], int)
            switches = np.atleast_1d(props[self.switch_key])
            unique_switches = np.unique(switches)
            try:
                max_p = unique_switches.max()
                min_p = unique_switches.min()
                assert max_p < self.n_elements
                assert min_p >= 0
            except AssertionError:
                raise ValueError(
                    f"{self.switch_key} must be ints of range [{min_p}, {max_p-1}]"
                )

            values = np.atleast_1d(switches).astype(np.float64)

            for i in unique_switches:
                el = self._elements[i]
                values_el = getattr(el, method_name)(props, **element_kwargs)
                np.copyto(values, values_el, casting="safe", where=(i == switches))
            return values[0] if was_int else values

        func.__doc__ = docs
        self.__setattr__(method_name, func)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        summary = super().get_summary()
        summary.update(
            {
                "switch_key": self.switch_key,
                "methods": self.methods,
                "n_elements": self.n_elements,
                "elements": [el.get_summary() for el in self.elements],
            }
        )
        return summary

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")
        if name is None:
            name = summary.pop("class")

        _ = summary.pop("elements")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        for el in self.elements:
            tree.add(el.tree)

        return tree

    def all_keys(self) -> list:
        """Get keys from all levels"""
        all_keys = self.keys()
        for el in self._elements:
            for key in el.keys():
                if key not in all_keys:
                    all_keys.append(key)
        return all_keys


class Blend(Element):
    """Blend `Element` types to create new hybrid Elements.

    The blending process is specific to the implementation which inherits Blend.

    Attributes:
        name (str): Name for switch
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods that the Switch should implement to match the Elements.
        n_elements (int): The number of elements
    """

    def __init__(
        self,
        blend_keys: List[str],
        elements: List[Type[Element]],
        methods: List[str],
        name=None,
    ):
        super().__init__(name=name, keys=blend_keys)
        self._blend_keys = blend_keys if isinstance(blend_keys, list) else [blend_keys]
        self._elements = elements
        self._methods = methods

        _element_check(self._elements, self._methods)

    def _check_props(self, props: Dict[str, NDArrayOrFloat]):
        for key in self._blend_keys:
            try:
                assert key in props
            except AssertionError:
                raise ValueError(f"Missing {key} from `props` input for {self.name}")

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def elements(self) -> List[Type[Element]]:
        return self._elements

    @property
    def methods(self) -> List[str]:
        return self._methods

    @property
    def blend_keys(self) -> List[str]:
        return self._blend_keys

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        summary = super().get_summary()
        summary.update(
            {
                "blend_keys": self.blend_keys,
                "methods": self.methods,
                "n_elements": self.n_elements,
                "elements": [el.get_summary() for el in self.elements],
            }
        )
        return summary

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")
        if name is None:
            name = summary.pop("class")

        _ = summary.pop("elements")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        for el in self.elements:
            tree.add(el.tree)

        return tree

    def all_keys(self) -> list:
        """Get keys from all levels"""
        all_keys = self.keys()
        for el in self._elements:
            for key in el.keys():
                if key not in all_keys:
                    all_keys.append(key)
        return all_keys
