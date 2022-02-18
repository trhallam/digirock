"""Mineral base class. Minerals are the end consumers for FrameModels.
"""

from typing import Tuple
from .._exceptions import WorkflowError
from .._base import Element

from ..elastic import acoustic_vel


class Mineral(Element):
    """Base Class for defining Minerals. They are rock frame constituents.

    Attributes:
        name (str): name of the mineral
        density (float): density of the mineral in g/cc
        k (float): Bulk modulus (compression) of the mineral in GPa
        mu (float): Shear modulus of the mineral in GPa
    """

    def __init__(
        self,
        density: float,
        bulk_modulus: float,
        shear_modulus: float,
        name: str = None,
    ):
        self.name = name
        self.density = density
        self.k = bulk_modulus
        self.mu = shear_modulus

    def _check_defined(self, from_func, var):
        if self.__getattribute__(var) is None:
            raise WorkflowError(from_func, f"The {var} attribute is not defined.")

    def elastic(self) -> Tuple[float, float, float]:
        """Pure elastic properties of mineral.

        Uses [`acoustic_vel`][digirock.elastic.acoustic_vel].


        Returns:
            compressional velocity (m/s), shear velocity (m/s), density (g/cc)
        """
        vp, vs = acoustic_vel(self.bulk_modulus, self.shear_modulus, self.density)
        return vp, vs, self.density

    @property
    def vp(self) -> float:
        """Compressional Velocity (m/s)"""
        vp, _, _ = self.elastic()
        return vp

    @property
    def vs(self) -> float:
        """Shear Velocity (m/s)"""
        _, vs, _ = self.elastic()
        return vs

    def get_summary(self) -> dict:
        summary = super().get_summary()
        vp, vs, dens = self.elastic()
        summary.update(
            {
                "name": self.name,
                "k": self.k,
                "mu": self.mu,
                "dens": dens,
                "vp": vp,
                "vs": vs,
            }
        )
        return summary
