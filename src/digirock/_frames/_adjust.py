"""Stress adjustment models. Usually this adjusts the porespace of your frame but can be
model specific.
"""
from typing import Dict, List, Tuple, Type, Sequence

from ..typing import NDArrayOrFloat
from ..utils._decorators import check_props
from .._base import Element, Transform
from .._exceptions import WorkflowError, PrototypeError
from .._stress import StressModel
from ..elastic import acoustic_vels, acoustic_velp
from ..models import dryframe_dpres


class StressAdjust(Transform):
    """Base porosity adjustment class

    Implements porosity adjustments for density, vp and vs.
    Classes which inherit this class should implement the modifications to the bulk and shear moduli.

    Attributes:

    """

    _methods = ["bulk_modulus", "vp", "vs", "shear_modulus", "density"]

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        stress_model: Type[StressModel],
        name: str = None,
    ):
        super().__init__(transform_keys, element, self._methods, name=name)
        self._stress_model = stress_model

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Applies the class porosity adjustment to the bulk modulus."""
        raise PrototypeError(self.__class__.__name__, "bulk_modulus")

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Applies the class porosity adjustment to the shear modulus."""
        raise PrototypeError(self.__class__.__name__, "shear_modulus")

    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns density of RockFrame using volume fraction average, see [mixed_denisty][digirock.models._mod.mixed_density].

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            density (g/cc)
        """
        dens = self.element.density(props, **kwargs)
        return dens * (1 - props["poro"])

    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns compression velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s).
        """
        density = self.density(props, **kwargs)
        bulk = self.bulk_modulus(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_velp(bulk, shear, density)

    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns shear velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s).
        """
        density = self.density(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_vels(shear, density)


class MacBethStressAdjustment(StressAdjust):
    """Adjustment model for porosity based upon MacBeth's stress response model.

    The properties of `pres_e` and `pres_ei` must be calculated using a StressModel.
    peffi = self.effective_stress(depth, pres1)
    #     peff = self.effective_stress(depth, pres2)


    Attributes:
        e_k (float): e exponent bulk modulus
        p_k (float): p exponent bulk modulus
        e_mu (float): e exponent shear modulus
        p_mu (float): p exponent shear modulus
    """

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        stress_model: Type[StressModel],
        e_k: float,
        p_k: float,
        e_mu: float,
        p_mu: float,
        name: str = None,
    ):
        super().__init__(transform_keys, element, stress_model, name=name)
        self.e_k = e_k
        self.p_k = p_k
        self.e_mu = e_mu
        self.p_mu = p_mu

    def _effective_pres(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Use the stress model to calculate effective pressure."""
        return self._stress_model.effective_stress(props, **kwargs)

    @check_props("pres_init")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """ """
        pres_eff = self._effective_pres(props, **kwargs)
        k0 = self.element.bulk_modulus(props, **kwargs)
        return dryframe_dpres(k0, props["pres_init"], pres_eff, self.e_k, self.p_k)

    @check_props("pres_init")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """ """
        pres_eff = self._effective_pres(props, **kwargs)
        mu0 = self.element.shear_modulus(props, **kwargs)
        return dryframe_dpres(mu0, props["pres_init"], pres_eff, self.e_mu, self.p_mu)
