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


class Water(Fluid):
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
        super().__init__(name=name)
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
        fvf1_pres (float): The reference pressure when the FVF=1.
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
        fvf1_pres: float = 0.101325,
    ):
        """
        Args:
            ref_pres: Reference pressure of bw, should be close to in-situ pressure (MPa).
            bw: Water formation volume factor at ref_pres (frac).
            comp: Compressibility of water at ref_pres (1/MPa)
            visc: Water viscosity at ref_pres (cP)
            cvisc: Water viscosibility (1/MPa)
            name (optional): Name for fluid. Defaults to None.
            salinity (optional): Salinity of brine (ppm). Defaults to 0.
            fvf1_pres (optional): The reference pressure when the FVF=1. Defaults to 0.101325 MPa.
        """
        super().__init__(name=name)
        self.sal = salinity / 1e6
        self.ref_pres = ref_pres
        self.bw = bw
        self.comp = comp
        self.visc = visc
        self.cvisc = cvisc
        self.fvf1_pres = fvf1_pres

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


class BaseOil(Fluid):
    """Base Oil Class for common methods.

    Attributes:
        name (str): name of the fluid
        api (float): API gravity of oil.
        std_density (float): Standard bulk density in g/cc at 15.6degC.
    """

    @mutually_exclusive("api", "std_density")
    def __init__(self, name: str = None, api: float = None, std_density: float = None):
        """
        `api` and `std_density` are mutually exclusive inputs.

        Args:
            name: Name of fluid
            api: Oil API
            std_density: Standard bulk density in g/cc at 15.6degC
        """
        super().__init__(name=name, keys=["bo"])
        if api is not None:
            self.set_api(api)
        elif std_density is not None:
            self.set_standard_density(std_density)
        else:
            self.api = None
            self.std_density = None

    def set_api(self, api: float):
        """Set the density of the oil using API gravity.

        Args:
            api: Api of oil.
        """
        self.api = api
        self.std_density = 141.5 / (self.api + 131.5)

    def set_standard_density(self, std_density: float):
        """Set the density of the oil at standard pressure and temperature (15.6 degC).

        Args:
            std_density: The density of oil at standard conditions (g/cc)
        """
        self.std_density = std_density
        self.api = 141.5 / self.std_density - 131.5


class DeadOil(BaseOil):
    """Dead Oil fluid class for oils with no dissolved gasses.

    Attributes:
        name (str): name of the fluid
        api (float): API gravity of oil.
        std_density (float): Standard bulk density in g/cc at 15.6degC.
        bo (float, xarray.DataArray): The formation volume factor or table.
    """

    @mutually_exclusive("api", "std_density")
    def __init__(self, name: str = None, api: float = None, std_density: float = None):
        """
        `api` and `std_density` are mutually exclusive inputs.

        Args:
            name: Name of fluid
            api: Oil API
            std_density: Standard bulk density in g/cc at 15.6degC
        """
        super().__init__(name=name, std_density=std_density, api=api)
        self.pvt = None

    def set_api(self, api: float):
        """Set the density of the oil using API gravity.

        Args:
            api: Api of oil.
        """
        self.api = api
        self.std_density = 141.5 / (self.api + 131.5)

    def set_standard_density(self, std_density: float):
        """Set the density of the oil at standard pressure and temperature (15.6 degC).

        Args:
            std_density: The density of oil at standard conditions (g/cc)
        """
        self.std_density = std_density
        self.api = 141.5 / self.std_density - 131.5

    def set_pvt(self, bo: NDArrayOrFloat, pres: NDArrayOrFloat = None):
        """Set the PVT table for DeadOil. If pressure is not
        specified bo should be a constant.

        Args:
            bo: Bo constant or table matching pressure input.
            pres: Pressures that bo is defined at. Defaults to None.

        Raises:
            ValueError: When inputs are incorrect.
        """
        if pres is None and isinstance(bo, (float, int)):
            self.pvt = {"type": "constant", "bo": bo}
        elif pres is None:
            raise ValueError(f"bo input is wrong type, got {type(bo)}")
        else:
            pres = np.array(pres).squeeze()
            bo = np.array(bo).squeeze()

            if bo.shape != pres.shape and bo.ndim > 1:
                raise ValueError("Expected 1d arrays of same length.")

            self.pvt = {
                "type": "table",
                "bo": "table",
                "pres": "table",
                "bo_table": xr.DataArray(bo, coords=[pres], dims=["pres"]),
            }

    def bo(self, pres: NDArrayOrFloat = None) -> NDArrayOrFloat:
        """Get the formation volume factor (bo) at pressure if specified.

        Args:
            pres: Pressure to sample bo (MPa), required when Bo is a table.

        Returns:
            Formation volume factor fvf (frac)
        """
        if self.pvt["type"] == "table" and pres is None:
            raise ValueError(
                "Bo is a table and the pressure argument must be specified."
            )

        if self.pvt["type"] == "constant":
            return self.pvt["bo"]
        else:
            return fluid_ecl.oil_fvf_table(
                self.pvt["bo_table"].pres.values, self.pvt["bo_table"].values, pres
            )

    @check_props("temp", "pres")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for dead oil adjusted for FVF.

        Uses BW92 [`oil_density`][digirock.fluids.bw92.oil_density].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)
            kwargs: ignored

        Returns:
            Oil density (g/cc)
        """
        return bw92.oil_density(self.std_density, props["pres"], props["temp"])

    @check_props("temp", "pres", broadcastable=("temp", "pres", "bo"))
    def velocity(
        self,
        props: Dict[str, NDArrayOrFloat],
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent acoustic velocity for dead oil adjusted for FVF.

        Uses BW92 [`oil_velocity`][digirock.fluids.bw92.oil_velocity], gas Rs is assumed to be 0.

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa); optional bo (vfrac) else use class constant or table.
            kwargs: ignored

        Returns:
            Oil acoustic velocity (m/s)
        """
        bo = props.get("bo")
        if bo is None:
            bo = self.bo(props["pres"])
        return bw92.oil_velocity(
            self.std_density, props["pres"], props["temp"], 0, 0, bo
        )

    def bulk_modulus(
        self,
        props: Dict[str, NDArrayOrFloat],
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for dead oil adjusted for FVF.

        Uses BW92 [`oil_bulkmod`][digirock.fluids.bw92.oil_bulkmod].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa); optional bo (vfrac) else use class constant or table.
            kwargs: ignored

        Returns:
            Oil modulus (GPa)
        """
        return bw92.oil_bulkmod(self.density(props), self.velocity(props))

    def get_summary(self) -> dict:
        if self._bo_istable:
            bo = "table"
        else:
            bo = self.bo
        summary = super().get_summary()
        summary.update(
            {"api": self.api, "std_density": self.std_density, "pvt": self.pvt}
        )
        return summary


class OilBW92(BaseOil):
    """Oil fluid class for oils with dissolved gas, i.e. Live Oil.

    Attributes:
        name (str): name of the fluid
        api (float): API gravity of oil.
        std_density (float): Standard bulk density in g/cc at 15.6degC.
        gas_sg (float): The dissolved gas standard gravity.
        pvt (dict, xarray.DataArray): PVT table for Oil
    """

    @mutually_exclusive("api", "std_density")
    def __init__(
        self,
        name: str = None,
        api: float = None,
        std_density: float = None,
        gas_sg: float = None,
    ):
        """
        `api` and `std_density` are mutually exclusive inputs.

        Args:
            name: Name of fluid
            api: Oil API
            std_density: Standard bulk density in g/cc at 15.6degC
        """
        self.gas_sg = gas_sg
        self.pvt = None
        super().__init__(name=name, api=api, std_density=std_density)
        self.register_key("rs")

    def set_pvt(
        self,
        rs: NDArrayOrFloat,
        pres: NDArrayOrFloat = None,
    ):
        """Set the PVT table for Oil.

        The solution gas ratio `rs` i[] set for tables of `bo` and `pres`.

        Args:
            rs: The solution gas ratio for bo or bo table. Has shape (M, )
            pres: Pressure values (MPa) to match rs if defined. Has shape (M, ).
        """
        rs = np.atleast_1d(rs)
        pres = np.atleast_1d(pres) if pres is not None else None

        # constants for rs and bo
        if rs.size == 1:
            self.pvt = {"type": "constant", "rs": rs[0], "pres": pres}
        elif len(rs.shape) > 1 and pres is None:
            raise ValueError("presure requires for list of `rs`")
        elif rs.shape != pres.shape:
            raise ValueError(
                f"`pres` {pres.shape} and `rs` {rs.shape} must have same shape"
            )
        elif rs.shape == pres.shape:
            table = xr.DataArray(data=rs, coords={"pres": pres})
            self.pvt = {
                "type": "table",
                "rs": "table",
                "pres": "table",
                "rs_table": table,
            }
        else:
            raise NotImplementedError("Unknown combination of rs and bo")

    def set_dissolved_gas(self, gas_sg: float):
        """Set the dissolved gas properties.

        Args:
            gas_sg: The dissolved gas specific gravity
        """
        self.gas_sg = gas_sg

    @broadcastable("temp", "pres")
    def _get_rsbo(
        self, temp: NDArrayOrFloat, pres: NDArrayOrFloat
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
        if self.pvt is None:
            raise WorkflowError(
                "RS/PRESSURE relationship needs to be set using `set_pvt()`"
            )

        if self.pvt["type"] == "constant":
            rs = self.pvt["rs"]
        elif self.pvt["type"] == "table":
            rs = self.pvt["rs_table"].interp(pres=pres).values
        else:
            raise NotImplementedError(f"PVT of type {self.pvt['type']} is unknown")

        if self.std_density is not None:
            fvf = bw92.oil_fvf(self.std_density, self.gas_sg, rs, temp)
        else:
            raise WorkflowError(
                "std_density", "Set an oil standard density or api first."
            )

        return rs, fvf

    def bo(self, temp: NDArrayOrFloat, pres: NDArrayOrFloat) -> NDArrayOrFloat:
        """Calculate the oil formation volume factor (bo) using BW92.

        Set the attribute `bo` using BW92 [oil_fvf][digirock.fluids.bw92.oil_fvf].

        Args:
            temp: The in-situ temperature (degC)
            pres: The pressure to calculate the gas `rs` for. Necessary if rs is specified as
                a table.
        """
        _, fvf = self._get_rsbo(temp, pres)
        return fvf

    def rs(self, pres: NDArrayOrFloat) -> NDArrayOrFloat:
        """Calculate the solution gas (rs) from pressure table.

        Args:
            pres: The pressure to calculate the gas `rs` for. Necessary if rs is specified as
                a table.
        """
        # temperatue is dummy
        rs, _ = self._get_rsbo(100.0, pres)
        return rs

    def _process_bo_rs(
        self, props: Dict[str, NDArrayOrFloat]
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
        # If RS or BO are not supplied, calculate from the table
        rs = props.get("rs")
        fvf = props.get("bo")
        if rs is None or fvf is None:
            rs_l, fvf_l = self._get_rsbo(props["temp"], props["pres"])
            return fvf_l if fvf is None else fvf, rs_l if rs is None else rs
        else:
            return fvf, rs

    @check_props("temp", "pres", broadcastable=("temp", "pres", "rs", "bo"))
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for Oil with adjustments for `rs` (solution gas) and `bo` (FVF).

        Density is calculated using BW92 [oil_density][digirock.fluids.bw92.oil_density] after adjusting for gas saturation with [oil_rho_sat][digirock.fluids.bw92.oil_rho_sat].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa); optional `bo` (vfrac) and `rs` (v/v) else use class consstant or table.
            kwargs: ignored

        Returns:
            Oil density (g/cc)
        """
        fvf, rs = self._process_bo_rs(props)
        oil_rho_s = bw92.oil_rho_sat(self.std_density, self.gas_sg, rs, fvf)
        oil_rho = bw92.oil_density(oil_rho_s, props["pres"], props["temp"])
        return oil_rho

    @check_props("temp", "pres", broadcastable=("temp", "pres", "rs", "bo"))
    def velocity(
        self,
        props: Dict[str, NDArrayOrFloat],
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent acoustic velocity for Oil adjusted for `rs` (solution gas) and `bo` (FVF).

        Velocity is calculated using BW92 [oil_velocity][digirock.fluids.bw92.oil_velocity].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa); optional `bo` (vfrac) and `rs` (v/v) else use class consstant or table.
            kwargs: ignored

        Returns:
            Oil velocity (m/s)
        """
        fvf, rs = self._process_bo_rs(props)
        return bw92.oil_velocity(
            self.std_density, props["pres"], props["temp"], self.gas_sg, rs, fvf
        )

    def bulk_modulus(
        self,
        props: Dict[str, NDArrayOrFloat],
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for Oil adjusted for `rs` (solution gas) and `bo` (FVF).

        Modulus is calculated using BW92 [oil_bulkmod][digirock.fluids.bw92.oil_bulkmod].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa); optional `bo` (vfrac) and
            kwargs: ignored

        Returns:
            Oil modulus (GPa)
        """
        return bw92.oil_bulkmod(
            self.density(props),
            self.velocity(props),
        )

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update({"pvt": self.pvt})
        return summary


class OilPVT(OilBW92):
    """Oil fluid class for oils with dissolved gas using PVT Tables, i.e. Live Oil.

    Uses BW92 elastic properties but replaces the `bo` calculation of BW92 with an
    explicit table.

    Attributes:
        name (str): name of the fluid
        api (float): API gravity of oil.
        std_density (float): Standard bulk density in g/cc at 15.6degC.
        gas_sg (float): The dissolved gas standard gravity.
        pvt (dict, xarray.DataArray): PVT table for Oil
    """

    @mutually_exclusive("api", "std_density")
    def __init__(
        self,
        name: str = None,
        api: float = None,
        std_density: float = None,
        gas_sg: float = None,
    ):
        """
        `api` and `std_density` are mutually exclusive inputs.

        Args:
            name: Name of fluid
            api: Oil API
            std_density: Standard bulk density in g/cc at 15.6degC
        """
        self.gas_sg = gas_sg
        self.pvt = None
        super().__init__(name=name, api=api, std_density=std_density)
        self.register_key("rs")

    def set_pvt(
        self,
        rs: NDArrayOrFloat,
        bo: NDArrayOrFloat,
        pres: NDArrayOrFloat = None,
    ):
        """Set the PVT table for Oil.

        The solution gas ratio `rs` i[] set for tables of `bo` and `pres`.

        Args:
            rs: The solution gas ratio for bo or bo table. Has shape (M, )
            bo: Can be a float of table of pressure (MPa) and bo (frac) pairs. Has shape (M, N)
            pres: Pressure values (MPa) to match bo. Has shape (M, N).
        """
        rs = np.atleast_1d(rs)
        bo = np.atleast_1d(bo)
        pres = np.atleast_1d(pres) if pres is not None else None

        # constants for rs and bo
        if rs.size == 1 and bo.size == 1:
            self.pvt = {"type": "constant", "rs": rs[0], "bo": bo[0], "pres": pres}

        # constant for rs and table for bo
        elif rs.size == 1 and bo.size > 1:
            if len(bo.shape) != 1:
                raise ValueError(
                    f"If `rs` is constant bo can only be 1d got shape {bo.shape}"
                )
            if pres is None:
                raise ValueError(
                    f"`pres` must be given for `bo` table with shape {bo.shape}"
                )
            elif pres.shape != bo.shape:
                raise ValueError(
                    f"`pres` {pres.shape} and `bo` {bo.shape} must have same shape"
                )

            table = xr.DataArray(data=np.arange(5), coords={"pres": np.arange(5) * 10})
            self.pvt = {
                "type": "fixed_rs",
                "rs": rs[0],
                "bo": "table",
                "pres": "table",
                "bo_table": table,
            }
            self.pvt.update(
                {
                    "bo_min": table.min().values,
                    "bo_max": table.max().values,
                    "pres_min": table["pres"].min().values,
                    "pres_max": table["pres"].max().values,
                }
            )

        # table for rs and table for bo
        elif rs.size > 1 and bo.size > 1:
            if len(rs.shape) > 1:
                raise ValueError(f"`rs` can at most be 1d got shape {rs.shape}")
            if len(bo.shape) != 2:
                raise ValueError(
                    f"If `rs` is a list bo can only be 2d got shape {bo.shape}"
                )
            if bo.shape[0] != rs.shape[0]:
                raise ValueError(
                    f"bo should have shape (M, N) for rs shape (M), got bo {bo.shape} and rs {rs.shape}"
                )
            if pres is None:
                raise ValueError(
                    f"`pres` must be given for `bo` table with shape {bo.shape}"
                )
            elif pres.shape != bo.shape:
                raise ValueError(
                    f"`pres` {pres.shape} and `bo` {bo.shape} must have same shape"
                )

            table = xr.concat(
                [
                    xr.DataArray(data=bo_i, coords={"pres": pres_i})
                    for bo_i, pres_i in zip(bo, pres)
                ],
                "rs",
            )
            table["rs"] = rs
            self.pvt = {
                "type": "full",
                "rs": "table",
                "bo": "table",
                "pres": "table",
                "bo_table": table,
            }
            self.pvt.update(
                {
                    "bo_min": table.min().values,
                    "bo_max": table.max().values,
                    "pres_min": table["pres"].min().values,
                    "pres_max": table["pres"].max().values,
                    "rs_min": table["rs"].min().values,
                    "rs_max": table["rs"].max().values,
                }
            )

        else:
            raise NotImplementedError("Unknown combination of rs and bo")

    def set_dissolved_gas(self, gas_sg: float):
        """Set the dissolved gas properties.

        Args:
            gas_sg: The dissolved gas specific gravity
        """
        self.gas_sg = gas_sg

    def calc_fvf(self, temp: float, pres: float):
        """Calculate the oil formation volume factor using BW92.

        Set the attribute `bo` using BW92 [oil_fvf][digirock.fluids.bw92.oil_fvf].

        Args:
            temp: The in-situ temperature (degC)
            pres: The pressure to calculate the gas `rs` for. Necessary if rs is specified as
                a table.
        """
        if self._rs_istable:
            rs = self.rs.interp(pres=pres)
        else:
            rs = self.rs

        if self.std_density is not None:
            self.bo = bw92.oil_fvf(self.std_density, self.gas_sg, rs, temp)
        else:
            raise WorkflowError(
                "std_density", "Set an oil standard density or api first."
            )

    def _get_rsfvf(self, pres: float):
        self._check_defined("fvf", "rs")
        self._check_defined("fvf", "bo")

        if self._rs_istable:
            rs = self.rs.interp(pres=pres).values
        else:
            rs = self.rs

        if self._bo_istable:
            if np.all(np.isnan(self.bo.pres.values)):
                fvf = self.bo.interp(rs=rs).values
            elif self.bo.rs.size == 1:
                fvf = self.bo.interp(pres=pres).values
            else:
                fvf = self.bo.interp(
                    rs=rs,
                    pres=pres,
                ).values
        else:
            fvf = self.bo
        return rs, fvf

    def fvf(self, pres: NDArrayOrFloat) -> NDArrayOrFloat:
        """Get the formation volume factor at pressure.
        TODO: Implement for different rs values

        Args:
            pres: Pressure to sample bo (MPa)

        Returns:
            Formation volume factor fvf (frac)
        """
        _, fvf = self._get_rsfvf(pres)
        return fvf


class Gas(Fluid):
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


class GasECL(Gas):
    """Gas fluid class which uses values from PVDG table and Density/Gravity Tables of ECL PROPS
    section.

    Attributes:
        name (str): name of the fluid
        density_asc: Density (g/cc) at surface conditions.
        pvt: list of pvt dicts, each table contains
            ref_pres: Reference pressure of bw (MPa)
            bw: Formation volume factor at ref_pres (rm3/sm3)
            comp: Compressibility of fluid. l/MPa
            visc: Viscosity of fluid cP
            cvisc: Viscosibility (1/MPa)
    """

    density_asc = None
    pvd = None
    _actpvtn = 0

    def __init__(self, name=None):
        """

        Args:
            name ([type], optional): [description]. Defaults to None.
        """
        super().__init__(name=name)

    def set_active_pvtn(self, pvtn):
        """Set the active PVT-N. To use for modelling."""
        # TODO: raise error if pvtn is larger than the number of tables in pvt
        self._actpvtn = int(pvtn)

    def density(self, temp, pres):
        """This function overides the density function in etlpy.pem.Fluid"""
        bg_func = interp1d(
            self.pvd[0]["pressure"],
            self.pvd[0]["bg"],
            bounds_error=False,
            fill_value="extrapolate",
        )

        bg = bg_func(pres)
        density = super().density(temp, pres)
        return density / bg

    def modulus(self, temp, pres):
        """This function overides the modulus function in etlpy.pem.Fluid
        Temperature argument is ignored assume const temp.
        """
        return pres / 100.0


class FluidModel:
    """Class for defining and building fluid models.

    Fluids are definied by adding component fluids to the model. Fluid properties can then be
    queried by submitting conditions.

    Attributes:
        name (str): Name of Fluid Model
        components (dict): component fluid properties
        ncomp (int): number of component fluids
    """

    def __init__(self, name=None, mixing_model="woods"):
        self.name = name
        self.components = dict()
        self._bdmcol.append("components")
        self.ncomp = 0
        self.mixing_model = mixing_model

    def add_component(self, name, fluid):
        """Add a fluid component to the model.

        Args:
            name (str): Name of fluid for key reference in other methods.
            fluid (etlpy.pem.Fluid): Derivative class of Fluid base class.
        """
        if not isinstance(fluid, Fluid):
            raise ValueError(f"fluid should be of type {Fluid} got {type(fluid)}")
        self.components[name] = fluid

    # def print fluids

    def _check_kwargs(self, vol_frac_tol=1e-3, **kwargs):
        if not kwargs:
            raise ValueError(f"Specify at least two fluids to mix.")

        lengths = []
        nonekey = None
        for key in kwargs:
            if key not in self.components.keys():
                raise ValueError(f"Unknown fluid keyword {key}")
            if kwargs[key] is None and nonekey is None:
                nonekey = key
            elif kwargs[key] is None and nonekey is not None:
                raise ValueError("Only one fluid component can be the complement")

            if not isinstance(kwargs[key], np.ndarray) and nonekey != key:
                lengths.append(1)
            elif nonekey != key:
                lengths.append(kwargs[key].size)

        n = len(lengths)
        if np.all(np.array(lengths) == lengths[0]):
            frac_test = np.zeros((lengths[0], n))
        else:
            raise ValueError(
                f"Input volume fractions must be the same size got {lengths}"
            )

        i = 0
        for key in kwargs:
            if key != nonekey:
                frac_test[:, i] = kwargs[key]
                i = i + 1

        if nonekey is not None:
            if np.all(frac_test.sum(axis=1) <= (1.0 + vol_frac_tol)):
                kwargs[nonekey] = 1 - frac_test.sum(axis=1)
            else:
                raise ValueError(
                    f"Input volume fractions sum to greater than 1"
                    + f" tolerance is {vol_frac_tol} and max sum was {np.max(frac_test.sum(axis=1))}"
                    + f" for keys {list(kwargs.keys())}"
                )
        else:
            if not np.allclose(frac_test.sum(axis=1), 1.0):
                raise ValueError(
                    f"Input volume fractions must sum to 1 if no complement."
                )

        return kwargs

    def _strip_oil_kwargs(self, kwargs):
        if "bo" in kwargs:
            bo = kwargs.pop("bo")
        else:
            bo = None

        if "rs" in kwargs:
            rs = kwargs.pop("rs")
        else:
            rs = None
        return kwargs, bo, rs

    def density(self, temp, pres, vol_frac_tol=1e-3, **kwargs):
        """Return the density of the mixed fluid.

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Volume fractions should sum to 1, pass a single fluid with value as None to set it
        as the complement.

        Args:
            kwargs: Volume fraction array.
        """
        # TODO: check temp and pres are good dims with kwargs

        kwargs, bo, rs = self._strip_oil_kwargs(kwargs)

        kwargs = self._check_kwargs(vol_frac_tol=vol_frac_tol, **kwargs)
        args = []
        for key, frac in kwargs.items():

            if isinstance(self.components[key], Oil):
                args = args + [
                    self.components[key].density(temp, pres, rs=rs, fvf=bo),
                    frac,
                ]
            else:
                args = args + [self.components[key].density(temp, pres), frac]
        return bw92.mixed_density(*args)

    def modulus(self, temp, pres, vol_frac_tol=1e-3, **kwargs):
        """Return the modulus of the mixed fluid.

        The kw arguments pass to this function are the volume fractions of each fluid name to mix.

        Volume frations should sum to 1, pass a single fluid with value as None to set it as the
        complement.

        Args:
            temp: Temperature for sameple point/s.
            pres: Pressure for sample point/s.
            kwargs: Volume fraction array for fluid kw.

        Returns:
           : The modulus of the mixed fluid for temp and pres points.
        """
        kwargs, bo, rs = self._strip_oil_kwargs(kwargs)
        kwargs = self._check_kwargs(vol_frac_tol=vol_frac_tol, **kwargs)
        args = []
        for key, frac in kwargs.items():
            if isinstance(self.components[key], Oil):
                args = args + [
                    self.components[key].density(temp, pres, rs=rs, fvf=bo),
                    frac,
                ]
            else:
                args = args + [self.components[key].density(temp, pres), frac]
        if self.mixing_model == "woods":
            return bw92.mixed_bulkmod(*args)

    def get_summary(self) -> dict:
        return {comp: self.components[comp].get_summary() for comp in self.components}
