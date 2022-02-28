"""Fluid base class to prototype generation of fluid properties.
"""
from typing import List, Dict, Type

from .._exceptions import PrototypeError, WorkflowError
from ..typing import NDArrayOrFloat
from .._base import Element, Switch


class Fluid(Element):
    """Base Class for defining fluids, all new fluids should be based upon this class.

    Attributes:
        name (str): name of the fluid
    """

    def __init__(self, name: str = None, keys: List[str] = None):
        Element.__init__(self, name, keys if keys else [])

    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns density of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Density for temp and pres (g/cc).
        """
        raise PrototypeError(self.__class__.__name__, "density")

    def velocity(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns acoustic velocity of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Velocity for temp and pres (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "velocity")

    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Alias for velocity"""
        return self.velocity(props, **kwargs)

    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Always returns 0"""
        return 0.0

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns bulk_modulus of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Modulus for temp and pres (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "modulus")

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Fluid shear modulus is zero. Return zero for all fluids.

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Modulus for temp and pres (GPa). Always 0.0
        """
        return 0.0

    def get_summary(self) -> dict:
        """Return a dictionary containing a summary of the fluid.

        Returns:
            Summary of properties.
        """
        return super().get_summary()


class FluidSwitch(Switch):
    """Class for fluid switching, e.g. when different fluid properties are needed in different PVT zones

    Implements the following [`Switch`][digirock.Switch] methods:

      - `density`
      - `bulk_modulus`
      - `shear_modulus`
      - `velocity`

    Attributes:
        name (str): Name for switch
        switch_key (str): Key to use for switching
        elements (list): A list of elements
        n_elements (int): The number of elements
    """

    _methods = ["density", "bulk_modulus", "shear_modulus", "velocity"]

    def __init__(self, switch_key: str, elements: List[Type[Fluid]], name=None):
        super().__init__(switch_key, elements, methods=self._methods, name=name)
