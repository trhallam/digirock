"""Fluid base class to prototype generation of fluid properties.
"""
from typing import List, Dict

from .._exceptions import PrototypeError, WorkflowError
from ..utils.types import NDArrayOrFloat
from .._base import BaseConsumerClass


class Fluid(BaseConsumerClass):
    """Base Class for defining fluids, all new fluids should be based upon this class.

    Attributes:
        name (str): name of the fluid
    """

    def __init__(self, name: str = None, keys: List[str] = None):
        BaseConsumerClass.__init__(self, name, keys if keys else [])

    def _check_defined(self, from_func, var):
        if self.__getattribute__(var) is None:
            raise WorkflowError(from_func, f"The {var} attribute is not defined.")

    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Returns density of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Density for temp and pres (g/cc).
        """
        raise PrototypeError(self.__class__.__name__, "density")

    def velocity(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Returns acoustic velocity of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Velocity for temp and pres (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "velocity")

    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
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
        # temp_ar = np.atleast_1d(temp)
        # temp_pres = np.atleast_1d(pres)
        return 0.0

    def get_summary(self) -> dict:
        """Return a dictionary containing a summary of the fluid.

        Returns:
            Summary of properties.
        """
        return super().get_summary()