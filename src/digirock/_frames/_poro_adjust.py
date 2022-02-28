"""RockFrame models to simplify generation of digitial rock frames.

"""

# pylint: disable=invalid-name,no-value-for-parameter
from typing import Dict, Sequence, Type, Any

import numpy as np

from ..typing import NDArrayOrFloat
from .._base import Transform, Element
from .._exceptions import PrototypeError, WorkflowError
from ..utils._decorators import check_props
from ..models import _mod
from ..elastic import acoustic_velp, acoustic_vels
from ..models._cemented_sand import dryframe_cemented_sand


class PoroAdjust(Transform):
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
        name: str = None,
    ):
        super().__init__(transform_keys, element, self._methods, name=name)

    @check_props("poro")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the bulk modulus."""
        raise PrototypeError(self.__class__.__name__, "bulk_modulus")

    @check_props("poro")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the shear modulus."""
        raise PrototypeError(self.__class__.__name__, "shear_modulus")

    @check_props("poro")
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


class FixedPoroAdjust(PoroAdjust):
    """Fixed Porosity Adjustment

    Modulii $m$ are transformed by porosity $\\phi$

    $$
    m_{\\phi} = m * (1-2.8 *\\phi)
    $$
    """

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        name: str = None,
    ):
        super().__init__(transform_keys, element, name=name)

    @check_props("poro")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the bulk modulus."""
        k0 = self.element.bulk_modulus(props, **kwargs)
        return (1 - 2.8 * props["poro"]) * k0

    @check_props("poro")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the shear modulus."""
        k0 = self.element.shear_modulus(props, **kwargs)
        return (1 - 2.8 * props["poro"]) * k0


class NurCriticalPoroAdjust(PoroAdjust):
    """Nur's Critical porosity adjustment."""

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        critical_poro: float,
        name: str = None,
    ):
        """

        Args:
            critical_poro: Critical porosity inflection point in Por vs Vp (0 < crit_por < 1)
        """
        super().__init__(transform_keys, element, name=name)
        if not (0 <= critical_poro <= 1):
            raise ValueError("Critical porosity should be 0 <= cp <= 1")
        self._critphi = critical_poro

    @property
    def critical_poro(self):
        return self._critphi

    def _tranform(self, poro: NDArrayOrFloat) -> NDArrayOrFloat:
        return np.where(poro >= self._critphi, 0.0, 1 - poro / self._critphi)

    @check_props("poro")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the bulk modulus."""
        k0 = self.element.bulk_modulus(props, **kwargs)
        k0_fact = self._tranform(props["poro"])
        return k0_fact * k0

    @check_props("poro")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the shear modulus."""
        mu0 = self.element.shear_modulus(props, **kwargs)
        mu0_fact = self._tranform(props["poro"])
        return mu0_fact * mu0

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        summary = super().get_summary()
        summary.update(
            {
                "critical_poro": self.critical_poro,
            }
        )
        return summary


class LeeConsPoroAdjust(PoroAdjust):
    """Lee/Pride 2005 Consolidation parameter porosity adjustment.

    Attributes:

    """

    def __init__(
        self,
        transform_keys: Sequence[str],
        element: Type[Element],
        cons_alpha: float,
        gamma: float = None,
        name: str = None,
    ):
        """

        If gamma is None:

        $$
        \\gamma = \\frac{1 + 2\\alpha}{1 + \\alpha}
        $$

        Args:

        """
        super().__init__(transform_keys, element, name=name)
        self._cons_alpha = cons_alpha
        self._cons_gamma = (
            (1 + 2 * cons_alpha) / (1 + cons_alpha) if gamma is None else gamma
        )

    @check_props("poro")
    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the bulk modulus.

        $$
        k_{dry} = k_0 * \\frac{1 - \\phi}{1 - \\alpha\\phi}
        $$
        """
        k0 = self.element.bulk_modulus(props, **kwargs)
        k0_fact = (1 - props["poro"]) / (1 + self._cons_alpha * props["poro"])
        return k0_fact * k0

    @check_props("poro")
    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Applies the class porosity adjustment to the shear modulus.

        $$
        \\mu_{dry} = \\mu_0\\frac{1 - \\phi}{1 - \\alpha\\gamma\\phi}
        $$
        """
        mu0 = self.element.shear_modulus(props, **kwargs)
        mu0_fact = (1 - props["poro"]) / (
            1 + self.cons_gamma * self._cons_alpha * props["poro"]
        )
        return mu0_fact * mu0

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of this class."""
        summary = super().get_summary()
        summary.update(
            {
                "critical_poro": self.critical_poro,
            }
        )
        return summary
