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
    """Base stress adjustment class

    Implements modulus adjustments for bulk modulus, shear modulus.
    Updates methods for density, vp and vs.

    Classes which inherit this class should implement the modifications to the bulk and shear moduli.

    Attributes:
        transform_keys tuple(str, ...): The props keys this model needs for performing the transform.
        element Element: The element to transform
        stress_model (StressModel): The stress model to use for effective stress
        name (str): Name of the model
    """

    _methods = ["bulk_modulus", "vp", "vs", "shear_modulus", "density"]

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        stress_model: Type[StressModel],
        name: str = None,
    ):
        """
        Args:
            transform_keys: The props keys this model needs for performing the transform.
            element: The element to transform
            stress_model: The stress model to use for effective stress
            name: Name of the model

        """
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
        return self.element.density(props, **kwargs)

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


class MacBethStressAdjust(StressAdjust):
    """Adjustment model for modulus based upon MacBeth's stress response model in a sandstone rock frame.

    $$
    \\kappa(P) = \\frac{\\kappa_\\inf}{1 + E_\\kappa e^{\\tfrac{-P}{P_\\kappa}}}
    $$

    $$
    \\mu(P) = \\frac{\\mu_\\inf}{1 + E_\\mu e^{\\tfrac{-P}{P_\\mu}}}
    $$

    This implementation removes estimates for $\\kappa_\\inf$ and $\\mu_\\inf$ by substituting in an initial
    pressure see [`dryframe_delta_pres`][digirock.models.dryframe_dpres].

    Attributes:
        e_k (float): e exponent bulk modulus
        p_k (float): p exponent bulk modulus
        e_mu (float): e exponent shear modulus
        p_mu (float): p exponent shear modulus

    Refs:
        MacBeth, C., 2004, A classification for the pressure-sensitivity properties of a sandstone rock frame, Geophysics, Vol 69, 2, pp497-510, doi:10.1190/1.1707070
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
        """
        Args:
            transform_keys: The props keys this model needs for performing the transform.
            element: The element to transform
            stress_model: The stress model to use for effective stress
            e_k: e exponent bulk modulus
            p_k: p exponent bulk modulus
            e_mu: e exponent shear modulus
            p_mu: p exponent shear modulus
            name: Name of the model

        """
        super().__init__(transform_keys, element, stress_model, name=name)
        self.e_k = e_k
        self.p_k = p_k
        self.e_mu = e_mu
        self.p_mu = p_mu

    def _effective_pres(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Use the stress model to calculate effective pressure."""
        return self._stress_model.effective_stress(props, **kwargs)

    def _initial_pres_props(
        self, props: Dict[str, NDArrayOrFloat], pres_key: str = "pres", **kwargs
    ):
        assert pres_key in props
        effi_props = {key: props[key] for key in self._stress_model.keys()}
        effi_props[pres_key] = props["pres_init"]
        return effi_props

    @check_props("pres_init")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """

        This implementation removes estimates for $\\kappa_\\inf$ by substituting in an initial
        pressure see [`dryframe_delta_pres`][digirock.models.dryframe_dpres].

        Args:
            props: props dict requires at least `pres` and `pres_init`.
            **kwargs: ignored

            Returns:
                bulk_modulus
        """
        pres_eff = self._effective_pres(props, **kwargs)
        effi_props = self._initial_pres_props(props, pres_key="pres", **kwargs)
        pres_effi = self._effective_pres(effi_props, **kwargs)
        k0 = self.element.bulk_modulus(props, **kwargs)
        return dryframe_dpres(k0, pres_effi, pres_eff, self.e_k, self.p_k)

    @check_props("pres_init")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """

        This implementation removes estimates for $\\mu_\\inf$ by substituting in an initial
        pressure see [`dryframe_delta_pres`][digirock.models.dryframe_dpres].

        Args:
            props: props dict requires at least `pres` and `pres_init`.
            **kwargs: ignored

        Returns:
            shear modulus
        """
        pres_eff = self._effective_pres(props, **kwargs)
        effi_props = self._initial_pres_props(props, pres_key="pres", **kwargs)
        pres_effi = self._effective_pres(effi_props, **kwargs)
        mu0 = self.element.shear_modulus(props, **kwargs)
        return dryframe_dpres(mu0, pres_effi, pres_eff, self.e_mu, self.p_mu)
