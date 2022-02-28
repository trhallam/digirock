"""Fluid models to simplify generation of fluid properties.

"""
from typing import List, Dict, Tuple, Type, Sequence, Any, Union

# pylint: disable=invalid-name,no-value-for-parameter
import numpy as np

from .._base import Blend, _get_complement, _volume_sum_check, Element
from ..typing import NDArrayOrFloat

from ..fluids import bw92

from ._fluid import Fluid
from ._oil import BaseOil


class WoodsFluid(Blend):
    """A Fluid blending Model that uses Wood's Style Mixing

    Attributes:
        name (str): Name for switch
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods that the Switch should implement to match the Elements.
        n_elements (int): The number of elements
        check_vol_sum (bool): Defaults to True, check volume fractions sum to 1.0
        vol_frac_tol (float): Apply an absolute tolerance 1.0 + `vol_frac_tol` to volume sum
    """

    _methods = ["density", "bulk_modulus", "shear_modulus", "velocity"]

    def __init__(
        self,
        vol_keys: List[str],
        elements: List[Type[Element]],
        name: str = None,
        check_vol_sum: bool = True,
        vol_frac_tol: float = 1e-3,
    ):
        """"""
        assert len(vol_keys) == len(elements)
        super().__init__(vol_keys, elements, self._methods, name=name)
        self.check_vol_sum = check_vol_sum
        self.vol_frac_tol = vol_frac_tol

    def density(self, props: Dict[str, NDArrayOrFloat], **element_kwargs):
        """Return the density of the mixed fluid based upon the volume fraction.

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Volume fractions should sum to 1, pass a single fluid with value as None to set it
        as the complement.

        Args:
            props: dictionary of properties, must contain all keys in `blend_keys`.
            element_kwargs: kwargs to pass to elements
        """
        # args = self._build_args(props, "density", **element_kwargs)
        args = self._process_props_get_method(props, "density", **element_kwargs)
        return bw92.mixed_density(*args)

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **element_kwargs
    ) -> NDArrayOrFloat:
        """Return the density of the mixed fluid based upon the volume fraction.

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Volume fractions should sum to 1, pass a single fluid with value as None to set it
        as the complement.

        Args:
            props: dictionary of properties, must contain all keys in `blend_keys`.
            element_kwargs: kwargs to pass to elements
        """
        args = self._process_props_get_method(props, "density", **element_kwargs)
        return bw92.woods_bulkmod(*args)

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **element_kwargs
    ) -> NDArrayOrFloat:
        """Return the density of the mixed fluid based upon the volume fraction.

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Always returns 0.0 (No Shear Modulus in fluids).

        Args:
            props: dictionary of properties, must contain all keys in `blend_keys`.
            element_kwargs: kwargs to pass to elements
        """
        return 0.0

    def velocity(
        self, props: Dict[str, NDArrayOrFloat], **element_kwargs
    ) -> NDArrayOrFloat:
        """Return the compressional velocity of the mixed fluid based upon the volume fraction by
        calculating the density and bulk modulus:

        $$
        v_p = \\sqrt{\\frac{\\kappa}{\\rho_b}}
        $$

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Volume fractions should sum to 1, pass a single fluid with value as None to set it
        as the complement.

        Args:
            props: dictionary of properties, must contain all keys in `blend_keys`.
            element_kwargs: kwargs to pass to elements
        """
        rhob = self.density(props, **element_kwargs)
        k = self.bulk_modulus(props, **element_kwargs)
        return np.sqrt(k / rhob) * 1000

    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Alias for velocity"""
        return self.velocity(props, **kwargs)

    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Always returns 0"""
        return 0.0
