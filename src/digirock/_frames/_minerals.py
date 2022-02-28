"""Mineral base class. Minerals are the end consumers for FrameModels.
"""
from typing import Tuple, Dict, Any

from ..typing import NDArrayOrFloat
from .._exceptions import WorkflowError
from .._base import Element

from ..elastic import acoustic_velp, acoustic_vels


class Mineral(Element):
    """Base Class for defining Minerals. They are rock frame constituents.

    Attributes:
        name (str): name of the mineral
        density (float): density of the mineral in g/cc
        bulk_modulus (float): Bulk modulus (compression) of the mineral in GPa
        shear_modulus (float): Shear modulus of the mineral in GPa
    """

    def __init__(
        self,
        density: float,
        bulk_modulus: float,
        shear_modulus: float,
        name: str = None,
    ):
        super().__init__(name=name)
        self._density = density
        self._bulk_modulus = bulk_modulus
        self._shear_modulus = shear_modulus

    def density(self, props: Any, **kwargs) -> float:
        """Return the density.

        Args:
            props: ignored
            kwargs: ignored

        Returns:
            constant value for Mineral density
        """
        return self._density

    def bulk_modulus(self, props: Any, **kwargs) -> float:
        """Return the bulk modulus.

        Args:
            props: ignored
            kwargs: ignored

        Returns:
            constant value for Mineral bulk modulus
        """
        return self._bulk_modulus

    def shear_modulus(self, props: Any, **kwargs) -> float:
        """Return the shear modulus.

        Args:
            props: ignored
            kwargs: ignored

        Returns:
            constant value for Mineral shear modulus
        """
        return self._shear_modulus

    def elastic(
        self, props: Any, **kwargs
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat, NDArrayOrFloat]:
        """Pure elastic properties of mineral.

        Uses [`acoustic_vel`][digirock.elastic.acoustic_vel].

        Returns:
            compressional velocity (m/s), shear velocity (m/s), density (g/cc)
        """
        return self.vp(None), self.vs(None), self.density(None)

    def vp(self, props: Any, **kwargs) -> NDArrayOrFloat:
        """Compressional Velocity (m/s)"""
        return acoustic_velp(self._bulk_modulus, self._shear_modulus, self._density)

    def vs(self, props: Any, **kwargs) -> NDArrayOrFloat:
        """Shear Velocity (m/s)"""
        return acoustic_vels(self._shear_modulus, self._density)

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update(
            {
                "name": self.name,
                "bulk_modulus": self.bulk_modulus(None),
                "shear_modulus": self.shear_modulus(None),
                "dens": self.density(None),
                "vp": self.vp(None),
                "vs": self.vs(None),
            }
        )
        return summary
