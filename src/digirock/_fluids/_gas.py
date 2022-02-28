"""Classes for Gas like Fluids"""

from typing import Dict, Tuple
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from ..utils._decorators import check_props, mutually_exclusive, broadcastable
from ..typing import NDArrayOrFloat
from ..fluids import bw92

from ._fluid import Fluid


class GasBW92(Fluid):
    """Gas fluid class.

    Notes that the acoustic velocity of gas is undefined.

    Attributes:
        name (str): name of the fluid
        gas_sg (float): Gas specific gravity.
        gas_density (float): Gas density (g/cc)
    """

    @mutually_exclusive("gas_sg", "gas_density")
    def __init__(
        self, name: str = None, gas_sg: float = None, gas_density: float = None
    ):
        """
        gas_sg and gas_density are mutually exclusive inputs.

        Args:
            name: Name of fluid
            gas_sg: Gas specific gravity
            gas_density: Gas density (g/cc)
        """
        super().__init__(name=name)
        if gas_density is not None:
            self.set_gas_density(gas_density)
        if gas_sg is not None:
            self.set_gas_sg(gas_sg)

    def set_gas_sg(self, gas_sg: float):
        """Set the gas sepcific gravity

        Calculates `gas_density` using BW92 density of air.

        Args:
            gas_sg: gas specific gravity
        """
        self.gas_sg = gas_sg
        self.gas_density = self.gas_sg * bw92.AIR_RHO

    def set_gas_density(self, gas_density: float):
        """Sets the gas density

        Calculates `gas_sg` using BW92 density of air.

        Args:
            gas_density: Density of gas
        """
        self.gas_density = gas_density
        self.gas_sg = self.gas_density / bw92.AIR_RHO

    @check_props("temp", "pres")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for Gas.

        Uses BW92 [`gas_oga_density`][digirock.fluids.bw92.gas_oga_density].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Gas density (g/cc)
        """
        return bw92.gas_oga_density(props["temp"], props["pres"], self.gas_sg)

    @check_props("temp", "pres")
    def bulk_modulus(
        self,
        props: Dict[str, NDArrayOrFloat],
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for Gas.

        Uses BW92 [`gas_adiabatic_bulkmod`][digirock.fluids.bw92.gas_adiabatic_bulkmod].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Oil modulus (GPa)
        """
        return bw92.gas_adiabatic_bulkmod(props["temp"], props["pres"], self.gas_sg)

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update({"density": self.gas_density, "sg": self.gas_sg})
        return summary


class GasPVT(Fluid):
    """Gas fluid class which uses values from PVDG table and Density/Gravity Tables of ECL PROPS
    section.

    Attributes:
        name (str): name of the fluid
        density_asc: Density (g/cc) at surface conditions.
        pvt: pvt dicts, contains
            ref_pres: Reference pressure of bw (MPa)
            bw: Formation volume factor at ref_pres (rm3/sm3)
            comp: Compressibility of fluid. l/MPa
            visc: Viscosity of fluid cP
            cvisc: Viscosibility (1/MPa)
    """

    def __init__(self, density_asc: float, name: str = None):
        """

        Args:
            density: The gas density at surface conditions (g/cc)
            name: The fluid ID
        """
        super().__init__(name=name)
        self.density_asc = density_asc
        self.register_key(density_asc)
        self.pvt = None
        self.register_key("temp")
        self.register_key("pres")

    def set_pvt(self, pres: NDArrayOrFloat, bg: NDArrayOrFloat):
        """Set the PVT table for the gas expansion for a given pressure.
        Args:
            pres: Pressure (MPa)
            bg: Expansion relative to surface conditions (vfrac)
        """
        pres = np.atleast_1d(pres)
        bg = np.atleast_1d(bg)
        try:
            assert len(pres.shape) == 1
        except AssertionError:
            raise ValueError("pres must be 1d")

        try:
            assert len(bg.shape) == 1
        except AssertionError:
            raise ValueError("bg must be 1d")

        try:
            assert bg.shape == pres.shape
        except AssertionError:
            raise ValueError("pres and bg must have same shape")

        self.pvt = {"pres": pres, "bg": bg}

        self._bg_func = interp1d(
            self.pvt["pres"],
            self.pvt["bg"],
            bounds_error=False,
            fill_value="extrapolate",
        )

    @check_props("pres")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Pressure dependent density for Gas.

        Uses BW92 [`gas_oga_density`][digirock.fluids.bw92.gas_oga_density] with
        bg from table.

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Gas density (g/cc)
        """
        bg = self._bg_func(props["pres"])
        return self.density_asc / bg

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update({"density_asc": self.density_asc, "pvt": self.pvt})
        return summary

    # def bulk_modulus(
    #     self,
    #     props: Dict[str, NDArrayOrFloat],
    #     **kwargs,
    # ) -> NDArrayOrFloat:
    #     """This function overides the modulus function in etlpy.pem.Fluid
    #     Temperature argument is ignored assume const temp.
    #     """
    #     return pres / 100.0
