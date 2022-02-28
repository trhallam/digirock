"""Mineral and Rock models to simplify generation of rock properties.

"""
# pylint: disable=invalid-name,bad-continuation
from typing import Dict, Union
import numpy as np

from ._exceptions import WorkflowError

from ._frames import RockFrame, PoroAdjust, StressAdjust, Mineral
from ._fluids import Fluid
from ._base import Transform, Blend, Switch
from .utils._decorators import check_props

from .typing import NDArrayOrFloat

from .elastic import acoustic_vel, acoustic_moduli, acoustic_velp, acoustic_vels
from .models import gassmann_fluidsub


class GassmannRock(Blend):
    """Build a rock model from minerals, fluids and methods which uses Gassmann
    fluid substitution.

    Attributes:
        name (str): name of the model

    """

    _methods = ["density", "vp", "vs", "shear_modulus", "bulk_modulus"]

    def __init__(
        self,
        frame_model: Union[RockFrame, PoroAdjust, StressAdjust, Mineral],
        zero_porosity_model: Union[RockFrame, Mineral],
        fluid_model: Union[Fluid, Blend, Switch],
        blend_keys=["poro"],
        name=None,
    ):
        """Gassmann rock model for fluid substitution into porespace.

        The frame model should be for a dry rock frame after any adjustments.

        Args:
            frame_model: dry rock frame model
            zero_porosity_model: need for k0 in Gassmann usually a VRHAvg or equivalent
            fluid_model: fluid model
            name: Name of the model
        """
        super().__init__(
            blend_keys,
            [frame_model, fluid_model, zero_porosity_model],
            methods=self._methods,
            name=name,
        )

    @check_props("poro")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """The density of the rock

        Args:
            props: Properties need for calculation e.g. porosity
            kwargs: passed to class elements

        Returns:
            density
        """
        # must be at least one fluid and one mineral component
        fluid_density = self.elements[1].density(props, **kwargs)
        min_dens = self.elements[0].density(props, **kwargs)
        return props["poro"] * fluid_density + min_dens

    @check_props("poro")
    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """The bulk modulus of the rock

        Args:
            props: Properties need for calculation e.g. porosity
            kwargs: passed to class elements

        Returns:
            Bulk modulus
        """
        kfl = self.elements[1].bulk_modulus(props, **kwargs)
        kdry = self.elements[0].bulk_modulus(props, **kwargs)
        k0 = self.elements[2].bulk_modulus(props, **kwargs)
        return gassmann_fluidsub(kdry, kfl, k0, props["poro"])

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Return the shear modulus of the rock

        Args:
            props: Properties need for calculation e.g. porosity
            kwargs: passed to class elements

        Returns:
            Shear modulus for input values
        """
        return self.elements[0].shear_modulus(props, **kwargs)

    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Compressional velocity of Gassmann Model

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s)
        """
        density = self.density(props, **kwargs)
        bulk = self.bulk_modulus(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_velp(bulk, shear, density)

    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Shear velocity of Gassmann Model

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s)
        """
        density = self.density(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_vels(shear, density)
