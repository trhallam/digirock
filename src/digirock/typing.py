from typing import Union, Dict
from numpy import ndarray
import pathlib

NDArrayOrFloat = Union[ndarray, float]
NDArrayOrInt = Union[ndarray, int]
Pathlike = Union[str, pathlib.Path]
PropsDict = Dict[str, Union[NDArrayOrFloat, NDArrayOrInt]]
