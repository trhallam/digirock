from typing import Union
from numpy import ndarray
import pathlib

NDArrayOrFloat = Union[ndarray, float]
NDArrayOrInt = Union[ndarray, int]
Pathlike = Union[str, pathlib.Path]
