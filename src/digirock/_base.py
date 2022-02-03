from typing import List


class BaseModelClass:

    _protected_kw_registry: list = list()

    def __init__(self, name: str = None, keys: List[str] = None):
        self.name = name
        if keys:
            for key in keys:
                try:
                    self.register_key(key)
                except ValueError:
                    pass  # ignore already registered keys if created using init

    @classmethod
    def register_key(self, key: str):
        """Register a new keyword across digirock classes"""
        if key in self._protected_kw_registry:
            raise ValueError(
                f"The key: {key}, cannot be registered more than once. Use another key."
            )
        self._protected_kw_registry.append(key)

    @classmethod
    def deregister_key(self, key: str):
        """Deregister a keyword across the digirock classes"""
        try:
            self._protected_kw_registry.remove(key)
        except ValueError:
            raise ValueError(
                f"The key: {key}, is not currently in the digirock registry."
            )

    @classmethod
    def keys(self):
        return self._protected_kw_registry

    def get_summary(self):
        return {"class": self.__class__}
