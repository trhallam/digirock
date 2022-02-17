"""Fluid models to simplify generation of fluid properties.

"""
from typing import List, Dict, Tuple

# pylint: disable=invalid-name,no-value-for-parameter
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

from ._exceptions import PrototypeError, WorkflowError
from .utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from .utils.ecl import EclStandardConditions, EclUnitMap, EclUnitScaler
from .utils._decorators import mutually_exclusive, check_props, broadcastable
from .utils.types import NDArrayOrFloat, Pathlike


from .fluids import bw92
from .fluids import ecl as fluid_ecl

from ._base import BaseConsumerClass


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
