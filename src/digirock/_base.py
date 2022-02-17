from functools import wraps, reduce
from typing import List, Dict
from more_itertools import all_equal

import numpy as np

from .utils.types import NDArrayOrFloat


class BaseConsumerClass:
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

    def keys(self) -> list:
        """Returns a list of keys this class will require for computation.

        Returns:
            The keys this class requires.
        """
        return self._protected_kw_registry

    def get_summary(self) -> dict:
        """Returns a summary of this class."""
        return {"class": self.__class__, "name": self.name, "props_keys": self.keys()}
