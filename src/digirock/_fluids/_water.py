"""Classes for Water like Fluids"""

from typing import Dict
import numpy as np

from ..utils.ecl import EclStandardConditions
from ..utils._decorators import check_props
from ..typing import NDArrayOrFloat
from ..fluids import bw92
from ..fluids import ecl as fluid_ecl

from ._fluid import Fluid


class WaterBW92(Fluid):
    """Water fluid class based upon B&W 92.

    Temperature (degC) and pressure (MPa) dependent.

    Attributes:
        name (str): name of the fluid
        sal (float): Brine salinity in ppm
    """

    def __init__(self, name: str = None, salinity: int = 0):
        """

        Args:
            name (optional): Name of fluid. Defaults to None.
            salinity (optional): Water salinity in ppm. Defaults to 0.
        """
        super().__init__(name=name, keys=["temp", "pres"])
        self.sal = salinity / 1e6

    @check_props("temp", "pres")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for water with fixed salt concentration.

        Uses BW92 [`wat_density_brine`][digirock.fluids.bw92.wat_density_brine].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            array-like: Water density (g/cc)
        """
        return bw92.wat_density_brine(props["temp"], props["pres"], self.sal)

    @check_props("temp", "pres")
    def velocity(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent acoustic velocity for water with fixed salt concentration.

        Uses BW92 [`wat_velocity_brine`][digirock.fluids.bw92.wat_velocity_brine].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Water velocity (m/s)
        """
        return bw92.wat_velocity_brine(props["temp"], props["pres"], self.sal)

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for water with fixed salt concentration.

        Uses BW92 [`wat_bulkmod`][digirock.fluids.bw92.wat_bulkmod].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Water modulus (GPa)
        """
        return bw92.wat_bulkmod(
            self.density(props),
            self.velocity(props),
        )

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update({"salinity": self.sal})
        return summary


class WaterECL(Fluid):
    """Water fluid class which uses Eclipse methodology for Density calculations.

    Modulus and velocity are calculated using B&W 92 with the modified density.

    Attributes:
        name (str): name of the fluid
        density_asc (float): Density (g/cc) at surface conditions.
        ref_pres (float): Reference pressure of bw (MPa)
        bw (float): Formation volume factor at ref_pres (rm3/sm3)
        comp (float): Compressibility of fluid. l/MPa
        visc (float): Viscosity of fluid cP
        cvisc (float): Viscosibility (1/MPa)
        salinity (float): Salinity in
        fvf1_pres (float): The reference pressure when the FVF=1. Defaults to 1Atm
    """

    def __init__(
        self,
        ref_pres: float,
        bw: float,
        comp: float,
        visc: float,
        cvisc: float,
        name: str = None,
        salinity: int = 0,
        fvf1_pres: float = None,
    ):
        """
        Args:
            ref_pres: Reference pressure of bw, should be close to in-situ pressure (MPa).
            bw: Water formation volume factor at ref_pres (frac).
            comp: Compressibility of water at ref_pres (1/MPa)
            visc: Water viscosity at ref_pres (cP)
            cvisc: Water viscosibility (1/MPa)
            name: Name for fluid. Defaults to None.
            salinity: Salinity of brine (ppm). Defaults to 0.
            fvf1_pres: The reference pressure when the FVF=1. Defaults to 0.101325 MPa.
        """
        super().__init__(name=name, keys=["temp", "pres"])
        self.sal = salinity / 1e6
        self.ref_pres = ref_pres
        self.bw = bw
        self.comp = comp
        self.visc = visc
        self.cvisc = cvisc
        self.fvf1_pres = fvf1_pres if fvf1_pres else EclStandardConditions["PRES"].value

    @property
    def density_asc(self) -> float:
        """Density at atmospheric conditions.

        Uses Batzle and Wang 92 with Eclipse Standard Condition values.

        Returns:
            Water density (g/cc)
        """
        return bw92.wat_density_brine(
            EclStandardConditions.TEMP.value, EclStandardConditions.PRES.value, self.sal
        )

    @check_props("temp", "pres")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for water with fixed salt concentration adjusted for FVF.

        Uses Eclipse [`e100_bw`][digirock.fluids.ecl.e100_bw] for calculating FVF. Eclipse multiplies the surface presure
        the expansion factor FVF relative to surface conditions to calculate the adjusted density.

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Water density (g/cc)
        """
        # pressure at atmospehric conditions i.e. fvf = 1
        bw_asc = fluid_ecl.e100_bw(
            self.fvf1_pres, self.ref_pres, self.bw, self.comp, self.visc, self.cvisc
        )
        bw = fluid_ecl.e100_bw(
            props["pres"], self.ref_pres, self.bw, self.comp, self.visc, self.cvisc
        )
        # density at atmospheric conditions
        return self.density_asc * bw_asc / bw

    @check_props("temp", "pres")
    def velocity(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent velocity for a fixed fixed salt concentration.

        Uses BW92 [`wat_velocity_brine`][digirock.fluids.bw92.wat_velocity_brine].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            velocity: Water velocity in m/s
        """
        return bw92.wat_velocity_brine(props["temp"], props["pres"], self.sal)

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for a fixed fixed salt concentration.

        Uses BW92 [`wat_bulkmod`][digirock.fluids.bw92.wat_bulkmod].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            modulus: Water modulus in GPa
        """
        return bw92.wat_bulkmod(
            self.density(props),
            self.velocity(props),
        )

    def get_summary(self):
        summary = super().get_summary()
        summary.update(
            {
                "salinity": self.sal,
                "ref_pres": self.ref_pres,
                "bw": self.bw,
                "comp": self.comp,
                "visc": self.visc,
                "cvisc": self.cvisc,
                "density_asc": self.density_asc,
            }
        )
        return summary
