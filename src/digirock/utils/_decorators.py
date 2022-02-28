"""Common decorators for functions.
"""
from typing import Dict, Callable, Sequence

from functools import wraps
import numpy as np
from inspect import getfullargspec

from ..typing import NDArrayOrFloat
from ._utils import check_broadcastable


def mutually_exclusive(keyword: str, *keywords: str) -> Callable:
    """Set keywords as being mutually exclusive

    Args:
        keyword: The first keyword that is exclusive.
        *keywords: The second or more keywords that are exclusive.

    Raises:
        TypeError: [description]

    Returns:
        mutually_exclusive checked function

    Ref:
        https://stackoverflow.com/a/54487188
    """
    keywords = (keyword,) + keywords

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if (
                kwargs
                and sum([k in keywords for k, val in kwargs.items() if val is not None])
                > 1
            ):
                raise ValueError(
                    "Expected no more than one of {}".format(", ".join(keywords))
                )
            return func(*args, **kwargs)

        return inner

    return wrapper


def mutually_inclusive(keyword, *keywords):
    """

    Args:
        keyword ([type]): [description]
    """
    keywords = (keyword,) + keywords

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            i = 0
            for k, val in kwargs.items():
                if isinstance(val, np.ndarray):
                    val = True
                if k in keywords and val not in (None, False):
                    i = i + 1
            if not (i == 0 or i == len(keywords)):
                raise ValueError(
                    "Expected none or all of keywords"
                    f"{', '.join(keywords)} to be defined."
                )
            return func(*args, **kwargs)

        return inner

    return wrapper


def broadcastable(keyword: str, *keywords: str) -> Callable:
    """Wrapper to check if the argument names listed are broadcastable as Numpy arrays.

    Args:
        keyword: argument to check
        keywords: other arguments to check against keyword

    Raises:
        ValueError: if the arrays cannot be broadcast
    """
    keywords = (keyword,) + keywords

    def wrapper(func):
        argspec = getfullargspec(func)

        @wraps(func)
        def inner(*args, **kwargs):
            # add args filtering to keywords check
            check_args = {
                name: arg for name, arg in zip(argspec.args, args) if name in keywords
            }
            # add keywords filtering to keywords check
            check_args.update({kw: val for kw, val in kwargs.items() if kw in keywords})
            _ = check_broadcastable(**check_args)
            return func(*args, **kwargs)

        return inner

    return wrapper


def check_props(
    *required_props: str, broadcastable: Sequence[str] = None, props_argument="props"
) -> Callable:
    """Wrapper to check props dictionary has required keywords.

    Args:
        required_props: strings of keys required in the props argument of the wrapped function
        broadcastable: strings in props that must be broadcastable if present, assumes all props if `broadcastable=None`
    """

    def wrapper(func):
        argspec = getfullargspec(func)
        assert props_argument in argspec.args

        @wraps(func)
        def inner(*args, **kwargs):
            props = [
                arg for arg, name in zip(args, argspec.args) if name == props_argument
            ][0]
            missing = [p for p in required_props if p not in props]
            if missing:
                raise ValueError(
                    f"{func} requires props kws: {required_props}, missing: {missing}"
                )
            if broadcastable is None:
                bc = tuple(props.keys())
            else:
                bc = broadcastable

            _ = check_broadcastable(**{name: props.get(name) for name in bc})
            return func(*args[:-1], props=props, **kwargs)

        return inner

    return wrapper
