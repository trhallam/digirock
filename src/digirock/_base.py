from typing import List, Dict, Sequence, Union, Type, Any
from rich.tree import Tree

import numpy as np

from .utils import check_broadcastable
from .utils._decorators import check_props
from .typing import NDArrayOrFloat


class Element:
    """Base consumer class from which all other consumer classes are defined.
    For example, uses to define input consumer classes like fluids, mineral mixing models etc.



    Attributes:
       name (str): The name of the Model.
    """

    def __init__(self, name: str = None, keys: Sequence[str] = None):
        """

        Args:
            name: The Element name/id, if None will be assigned an instance ID
            keys: Keys this element will consume, keys must be unique
        """
        self._protected_kw_registry = list()
        self.name = name if name is not None else f"{type(self).__name__}_{id(self)}"
        if keys:
            for key in keys:
                self.register_key(key)

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

    def all_keys(self) -> List[str]:
        return self.keys()

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        return {"class": self.__class__, "name": self.name, "props_keys": self.keys()}

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        return tree

    def trace(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        """Returns a props trace for all methods, switching keys in props are ignored.

        Args:
            props:
            methods:
            **kwargs: passed to methods

        Returns:
            trace of method values through model tree
        """
        if isinstance(methods, str):
            methods = [methods]

        trace = {
            meth: getattr(self, meth, lambda props, **kwargs: None)(props, **kwargs)
            for meth in methods
        }
        trace["name"] = self.name
        return trace

    def trace_tree(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Tree:
        """Returns a props trace for all methods, switching keys in props are ignored.

        Args:
            props:
            methods:
            **kwargs: passed to methods

        Returns:
            tree view of the trace
        """
        trace = self.trace(props, methods, **kwargs)
        name = trace.pop("name")
        t = Tree(name)
        _trace_tree(trace, t)

        return t


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


def _trace_tree(trace: dict, tree: Tree) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename

    sub_items = []
    for key, item in trace.items():
        if isinstance(item, dict):
            sub_items.append((key, item))
        else:
            tree.add(f"{key} : {item}")

    for key, item in sub_items:
        try:
            name = item.pop("name")
        except KeyError:  # probably an attribute
            name = key
        branch = tree.add(name)
        _trace_tree(item, branch)

    # for key, item in trace.items():
    #     if isinstance(item, dict):
    #         branch = tree.add(key)
    #         _trace_tree(item, branch)
    #     else:
    #         tree.add(f"{key} : {item}")


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
                "elements": {
                    f"element{i}": el.get_summary()
                    for i, el in enumerate(self.elements)
                },
            }
        )
        return summary

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")

        _ = summary.pop("elements")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        for el in self.elements:
            tree.add(el.tree)

        return tree

    def trace(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        """Returns a props trace for all methods, switching keys in props are ignored.

        Args:
            props:
            methods:
            **kwargs: passed to methods

        Returns:
            trace of method values through model tree
        """
        if isinstance(methods, str):
            methods = [methods]

        switch_trace = {
            self.name: {
                el.name: el.trace(props, methods, **kwargs) for el in self._elements
            }
        }
        return switch_trace

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
        blend_keys (list): Keys to use for blending (must be unique)
        elements (list): A list of elements
        methods (list): A list of methods that the Switch should implement to match the Elements.
        n_elements (int): The number of elements
    """

    def __init__(
        self,
        blend_keys: Sequence[str],
        elements: Sequence[Type[Element]],
        methods: Sequence[str],
        name=None,
    ):
        super().__init__(name=name, keys=blend_keys)
        self._blend_keys = blend_keys if isinstance(blend_keys, list) else [blend_keys]
        self._elements = tuple(elements)
        self._methods = tuple(methods)

        _element_check(self._elements, self._methods)

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
                "elements": {
                    f"element{i}": el.get_summary()
                    for i, el in enumerate(self.elements)
                },
            }
        )
        return summary

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")

        _ = summary.pop("elements")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        for el in self.elements:
            sum = el.get_summary()
            name = sum.pop("name")
            branch = tree.add(name)
            _trace_tree(sum, branch)

        return tree

    def trace(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        """Returns a props trace for all methods, switching keys in props are ignored.

        Args:
            props:
            methods:
            **kwargs: passed to methods

        Returns:
            trace of method values through model tree
        """
        if isinstance(methods, str):
            methods = [methods]

        blend_trace = {
            el.name: el.trace(props, methods, **kwargs) for el in self._elements
        }
        blend_trace["name"] = self.name

        for meth in methods:
            blend_trace[meth] = getattr(self, meth, None)(props, **kwargs)
        return blend_trace

    def all_keys(self) -> list:
        """Get keys from all levels"""
        all_keys = self.keys()
        for el in self._elements:
            for key in el.all_keys():
                if key not in all_keys:
                    all_keys.append(key)
        return all_keys

    def _process_props_get_method(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Sequence[NDArrayOrFloat]:
        """Process the props to find if all required keys are present for blending

        Uses self.blend_keys for check and get the result of method by passing props.
        Will find the complement if a blend key is missing (assumes that the volumes
        must sum to 1.)

        Args:
            props
            methods


        Returns:
            result for method from elements in order of `blend_keys`
        """
        missing = [key for key in self.blend_keys if key not in props]
        if len(missing) > 1:
            raise ValueError(
                f"Had {len(missing)} missing volume fractions, only 1 missing volume fraction allowed: please add to props {missing}"
            )
        has_keys = [key for key in self.blend_keys if key in props]

        args = []

        if isinstance(methods, str):
            methods = [methods]

        for key in has_keys:
            eli = self.blend_keys.index(key)
            args += [
                getattr(self._elements[eli], meth)(props, **kwargs) for meth in methods
            ] + [props[key]]

        if missing:
            eli = self.blend_keys.index(missing[0])
            args += [
                getattr(self._elements[eli], meth)(props, **kwargs) for meth in methods
            ]
        return tuple(args)


class Transform(Element):
    """Transform `Element` types to adjust the outputs of those Elements.

    The transform process is specific to the implementation which inherits Transform.

    Attributes:
        name (str): Name for switch
        transform_keys (list): Keys to use for transforming (must be unique)
        element: The element to transform
        methods (list): A list of methods that the Transform should implement to match the Element.
    """

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        methods: Sequence[str],
        name=None,
    ):
        super().__init__(name=name, keys=transform_keys)
        self._transform_keys = (
            transform_keys if isinstance(transform_keys, list) else [transform_keys]
        )
        self._element = element
        self._methods = tuple(methods)

        _element_check([self._element], self._methods)

    @property
    def element(self) -> Type[Element]:
        return self._element

    @property
    def elements(self) -> List[Type[Element]]:
        return [self._element]

    @property
    def methods(self) -> List[str]:
        return self._methods

    @property
    def transform_keys(self) -> List[str]:
        return self._transform_keys

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif attr in self.methods:
            return getattr(self._element, attr)
        else:
            raise AttributeError(f"{attr} not available")

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        summary = super().get_summary()
        summary.update(
            {
                "transform_keys": self.transform_keys,
                "methods": self.methods,
                "elements": {
                    f"element{i}": el.get_summary()
                    for i, el in enumerate(self.elements)
                },
            }
        )
        return summary

    @property
    def tree(self) -> Tree:
        """Prints a rich tree view of the Class"""
        summary = self.get_summary()
        name = summary.pop("name")

        _ = summary.pop("elements")

        tree = Tree(str(name))
        for var, val in summary.items():
            tree.add(f"{var} : {val}")

        for el in self.elements:
            sum = el.get_summary()
            name = sum.pop("name")
            branch = tree.add(name)
            _trace_tree(sum, branch)

        return tree

    def trace(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Dict[str, Any]:
        """Returns a props trace for all methods, switching keys in props are ignored.

        Args:
            props:
            methods:
            **kwargs: passed to methods

        Returns:
            trace of method values through model tree
        """
        if isinstance(methods, str):
            methods = [methods]

        tsfm_trace = {
            el.name: el.trace(props, methods, **kwargs) for el in self.elements
        }
        tsfm_trace["name"] = self.name

        for meth in methods:
            tsfm_trace[meth] = getattr(self, meth, None)(props, **kwargs)
        return tsfm_trace

    def all_keys(self) -> list:
        """Get keys from all levels"""
        all_keys = self.keys()
        for el in self.elements:
            for key in el.keys():
                if key not in all_keys:
                    all_keys.append(key)
        return all_keys
