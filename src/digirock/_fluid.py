"""Fluid models to simplify generation of fluid properties.

"""
from pathlib import Path
from typing import List, Dict

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

    def density(self, props: Dict[str, NDArrayOrFloat] = None, **kwargs):
        """Returns density of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Density for temp and pres (g/cc).
        """
        raise PrototypeError(self.__class__.__name__, "density")

    def velocity(self, props: Dict[str, NDArrayOrFloat] = None, **kwargs):
        """Returns acoustic velocity of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Velocity for temp and pres (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "velocity")

    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat] = None, **kwargs):
        """Returns bulk_modulus of fluid

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Modulus for temp and pres (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "modulus")

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
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
    def density(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
    ) -> NDArrayOrFloat:
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
    def velocity(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
    ) -> NDArrayOrFloat:
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
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
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
    def density(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
    ) -> NDArrayOrFloat:
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
    def velocity(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
    ) -> NDArrayOrFloat:
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
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
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


def load_pvtw(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvtw",
    salinity: List[int] = None,
) -> dict:
    """Load a PVTW table into multiple [`WaterECL`][digirock.WaterECL] classes.

    PVTW tables have the form (where units may differ):

    ```
    PVTW
    --RefPres        Bw          Cw           Vw         dVw
    --   bara       rm3/m3       1/bara        cP        1/bara
        268.5      1.03382    0.31289E-04   0.38509    0.97801E-04 / --  #1
        268.5      1.03382    0.31289E-04   0.38509    0.97801E-04 / --  #2
    ```

    If salinity is None, the salinity is backed out from the Water density using BW92
    [`wat_salinity_brine`][digirock.fluids.bw92.wat_salinity_brine]. the DENSITY keyword must
    be present in the same input file.

    Args:
        filepath: Filepath or str to text file containing PVTW keyword tables.
        units: The Eclipse units of the PVTW table, one of ['METRIC', 'PVTM', 'FIELD', 'LAB']. Defaults to 'METRIC'.
        prefix: The prefix to apply to the name of each loaded fluid.
        salinity: The salinity of the pvt tables in PPM

    Returns:
        A dictionary of WaterECL instance for each PVT table entry.

    Raises:
        AssertionError: units is not valid
        ValueError: DENSITY kw not in file with PVTW
    """
    assert units in EclUnitMap.__members__

    table = dict()
    _ut: dict = EclUnitScaler[units].value

    rawpvt = read_eclipsekw_2dtable(filepath, "PVTW")
    dens = None

    try:
        dens = read_eclipsekw_2dtable(filepath, "DENSITY")
        dens = [_ut["density"] * float(d[1]) for d in dens]
        salinity = list(
            map(
                lambda x: bw92.wat_salinity_brine(
                    EclStandardConditions.TEMP.value,
                    EclStandardConditions.PRES.value,
                    x,
                ),
                dens,
            )
        )

    except KeyError:
        grav = read_eclipsekw_2dtable(filepath, "GRAVITY")
        raise NotImplementedError(
            "GRAVITY KW not yet implemented contact via Github for help."
        )
        # convert to density

    if dens is None and salinity is None:
        raise ValueError(
            "Require either DENSITY kw in input file or user specified salinity."
        )

    salinity_ar = np.atleast_1d(salinity)
    if salinity_ar.size == 1:
        salinity = np.full(len(rawpvt), salinity[0])

    for i, (rawtab, sal) in enumerate(zip(rawpvt, salinity_ar)):
        name = f"{prefix}{i}"
        tab = [
            _ut[units] * float(val)
            for val, units, name in zip(
                rawtab,
                ["pressure", "unitless", "ipressure", "unitless", "ipressure"],
                ["ref_pres", "bw", "comp", "visc", "cvisc"],
            )
        ]
        table[name] = WaterECL(*tab, name=name, salinity=sal * 1e6)

    return table


class DeadOil(Fluid):
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
        super().__init__(name=name, keys=["bo"])
        if api is not None:
            self.set_api(api)
        elif std_density is not None:
            self.set_standard_density(std_density)
        else:
            self.api = None
            self.std_density = None

        self.bo = None
        self._bo_istable = False

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

    def _fvf_array_from_table(self, pres, bo):
        self.bo = xr.DataArray(bo, coords=[pres], dims=["pres"])
        self._bo_istable = True

    def set_fvf(self, bo: NDArrayOrFloat, pres: NDArrayOrFloat = None):
        """Set the FVF factor relationship for the oil. If pressure is not
        specified bo should be a constant.

        Args:
            bo: Bo constant or table matching pressure input.
            pres: Pressures that bo is defined at. Defaults to None.

        Raises:
            ValueError: When inputs are incorrect.
        """
        if pres is None and isinstance(bo, (float, int)):
            self.bo = bo
        elif pres is None:
            raise ValueError(f"bo input is wrong type, got {type(bo)}")
        else:
            pres = np.array(pres).squeeze()
            bo = np.array(bo).squeeze()

            if bo.shape != pres.shape and bo.ndim > 1:
                raise ValueError("Expected 1d arrays of same length.")

            self._fvf_array_from_table(pres, bo)

    def fvf(self, pres: NDArrayOrFloat = None) -> NDArrayOrFloat:
        """Get the formation volume factor at pressure if specified.

        Args:
            pres: Pressure to sample bo (MPa), required when Bo is a table.

        Returns:
            Formation volume factor fvf (frac)
        """
        if self._bo_istable and pres is None:
            raise ValueError(
                "Bo is a table and the pressure argument must be specified."
            )

        if not self._bo_istable:
            return self.bo
        else:
            return fluid_ecl.oil_fvf_table(self.bo.pres.values, self.bo.values, pres)

    def calc_fvf(self, temp: float):
        """Calculate the oil formation volume factor using BW92.

        Uses [`oil_fvf`][digirock.fluids.bw92.oil_fvf] gas Rs is assumed 0.

        Args:
            temp (float): The in-situ temperature degC
        """
        if self.std_density is not None:
            self.bo = bw92.oil_fvf(self.std_density, 0, 0, temp)
        else:
            raise WorkflowError(
                "std_density", "Set an oil standard density or api first."
            )

    @check_props("temp", "pres")
    def density(
        self, props: Dict[str, NDArrayOrFloat] = None, **kwargs
    ) -> NDArrayOrFloat:
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
        props: Dict[str, NDArrayOrFloat] = None,
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
            bo = self.fvf(props["pres"])
        return bw92.oil_velocity(
            self.std_density, props["pres"], props["temp"], 0, 0, bo
        )

    def bulk_modulus(
        self,
        props: Dict[str, NDArrayOrFloat] = None,
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
        summary.update({"api": self.api, "std_density": self.std_density, "bo": bo})
        return summary


class Oil(DeadOil):
    """Oil fluid class for oils with dissolved gas, i.e. Live Oil.

    Attributes:
        name (str): name of the fluid
        api (float): API gravity of oil.
        std_density (float): Standard bulk density in g/cc at 15.6degC.
        bo (float, xarray.DataArray): The formation volume factor or table.
        gas_sg (float): The dissolved gas standard gravity.
        rs (float, xarray.DataArray): The dissolved gas ratio or table.
        sal (float): Brine salinity in ppm
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
        self.gas_sg = None
        self.rs = None
        self._rs_istable = False
        super().__init__(name=name, api=api, std_density=std_density, keys=["rs", "bo"])

    def _fvf_array_from_table(self, pres: NDArrayOrFloat, bo, rs, append=True):
        if not isinstance(rs, (int, float)):
            raise ValueError("rs should be a single value of type (float, int)")

        if not (bo.ndim == 1 and pres.ndim == 1):
            raise ValueError("bo and pres must be 1D arrays")

        if not isinstance(self.bo, xr.DataArray) or not append:
            self.bo = xr.DataArray(
                np.expand_dims(bo, 0), coords=[[rs], pres], dims=["rs", "pres"]
            )
        else:
            self.bo = self.bo.combine_first(
                xr.DataArray(
                    np.expand_dims(bo, 0), coords=[[rs], pres], dims=["rs", "pres"]
                )
            )

    def set_fvf(
        self,
        bo: NDArrayOrFloat,
        rs: float,
        pres: NDArrayOrFloat = None,
        append: bool = True,
    ):
        """Set the formation volume factor for fluid.

        Multiple values of `bo` for different `rs` values can be specified using the
        append keyword to create a multi-dimensional table.

        Args:
            bo: Can be a float of table of pressure (MPa) and bo (frac) pairs.
            rs: The solution gas ratio for bo or bo table.
            pres: Pressure values (MPa) to match bo
            append: Append to the existing fvf table with another `rs` point.
        """
        try:
            n = bo.size
        except AttributeError:
            n = 1

        if pres is None:
            pres = np.full(n, np.nan)

        self._bo_istable = True
        if isinstance(bo, np.ndarray) and isinstance(pres, np.ndarray):
            self._fvf_array_from_table(pres, bo, rs=rs, append=append)
        elif isinstance(bo, np.ndarray) and pres is None:
            raise ValueError("pres array required for bo array input")
        elif isinstance(bo, (int, float)):
            self._fvf_array_from_table(pres, np.r_[bo], rs=rs, append=append)
        else:
            raise ValueError(f"Unknow type for bo {type(bo)}")

    def _rs_array_from_table(self, pres: NDArrayOrFloat, rs: NDArrayOrFloat):

        if not (rs.ndim == 1 and pres.ndim == 1):
            raise ValueError("rs and pres must be 1D arrays")

        self.rs = xr.DataArray(rs, coords=[pres], dims=["pres"])

    def set_dissolved_gas(
        self, gas_sg: float, rs: NDArrayOrFloat, pres: NDArrayOrFloat = None
    ):
        """Set the dissolved gas properties.

        Args:
            gas_sg: The dissolved gas specific gravity
            rs: The gas to oil ratio (vfrac)
            pres: Required if `rs` is NDArray
        """
        if pres is None:
            try:
                _ = rs.size
                raise ValueError("pres must be provided if rs is array-like")
            except AttributeError:
                pass

        self.gas_sg = gas_sg
        self._rs_istable = True
        if isinstance(rs, np.ndarray):
            self._rs_array_from_table(pres, rs)
        elif isinstance(rs, (int, float)):
            self.rs = rs
            self._rs_istable = False
        else:
            raise ValueError(f"Unknown type for rs {type(rs)}")

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

    @check_props("temp", "pres")
    def density(
        self,
        temp: NDArrayOrFloat,
        pres: NDArrayOrFloat,
        rs: NDArrayOrFloat = None,
        fvf: NDArrayOrFloat = None,
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent density for Oil with adjusted for RS and FVF.

        Density is calculated using BW92 [oil_density][digirock.fluids.bw92.oil_density].

        Args:
            temp: Temperature in degC
            pres: Pressure in MPa
            rs: If None, uses set values or table
            fvf: If None, uses set values or table
            kwargs: ignored

        Returns:
            Oil density (g/cc)
        """
        if rs is None and fvf is None:
            rs, fvf = self._get_rsfvf(pres)
        elif rs is None and fvf is not None:
            rs, _ = self._get_rsfvf(pres)
        elif rs is not None and fvf is None:
            _, fvf = self._get_rsfvf(pres)

        oil_rho_s = bw92.oil_rho_sat(self.std_density, self.gas_sg, rs, fvf)
        oil_rho = bw92.oil_density(oil_rho_s, pres, temp)
        return oil_rho

    @check_props("temp", "pres")
    def velocity(
        self,
        temp: NDArrayOrFloat,
        pres: NDArrayOrFloat,
        rs: NDArrayOrFloat = None,
        fvf: NDArrayOrFloat = None,
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent acoustic velocity for Oil with adjusted for RS and FVF.

        Velocity is calculated using BW92 [oil_velocity][digirock.fluids.bw92.oil_velocity].

        Args:
            temp: Temperature in degC
            pres: Pressure in MPa
            rs: If None, uses set values or table
            fvf: If None, uses set values or table
            kwargs: ignored

        Returns:
            Oil velocity (m/s)
        """
        if rs is None and fvf is None:
            rs, fvf = self._get_rsfvf(pres)
        elif rs is None and fvf is not None:
            rs, _ = self._get_rsfvf(pres)
        elif rs is not None and fvf is None:
            _, fvf = self._get_rsfvf(pres)

        return bw92.oil_velocity(self.std_density, pres, temp, self.gas_sg, rs, fvf)

    @check_props("temp", "pres")
    def bulk_modulus(
        self,
        temp: NDArrayOrFloat,
        pres: NDArrayOrFloat,
        rs: NDArrayOrFloat = None,
        fvf: NDArrayOrFloat = None,
        **kwargs,
    ) -> NDArrayOrFloat:
        """Temperature and pressure dependent bulk modulus for Oil with adjusted for RS and FVF.

        Modulus is calculated using BW92 [oil_bulkmod][digirock.fluids.bw92.oil_bulkmod].

        Args:
            temp: Temperature in degC
            pres: Pressure in MPa
            rs: If None, uses set values or table
            fvf: If None, uses set values or table
            kwargs: ignored

        Returns:
            Oil modulus (GPa)
        """
        return bw92.oil_bulkmod(
            self.density(temp, pres, rs=rs, fvf=fvf),
            self.velocity(temp, pres, rs=rs, fvf=fvf),
        )

    def get_summary(self):
        if self._rs_istable:
            rs = "table"
        else:
            rs = self.rs
        summary = super().get_summary()
        summary.update({"rs": rs})
        return summary


def load_pvto(filepath: Pathlike, units: str = "METRIC", prefix: str = "pvto") -> dict:
    """Load a PVTO Eclipse keyword from a textfile into multiple [`Oil`][digirock.Oil] classes.

    Args:
        filepath: The filepath of the file text file containing the PVTO tables.
        units: The Eclipse units of the PVTW table, one of ['METRIC', 'PVTM', 'FIELD', 'LAB']. Defaults to 'METRIC'.
        prefix: The prefix to apply to the name of each loaded fluid.

    Raises:
        AssertionError: units is not valid
    """
    assert units in EclUnitMap.__members__
    pvt = read_eclipsekw_3dtable(filepath, "PVTO")

    pvt_float = []
    for p in pvt:
        q_float = []
        for q in p:
            q_float = q_float + [np.array(q.split()).astype(float)]
        pvt_float = pvt_float + [q_float]

    pvt_rs_float = []
    for p in pvt_float:
        rs_float = []
        q_float = []
        for q in p:
            try:
                rs_float.append(q[0])
            except IndexError:
                continue
            q_float = q_float + [q[1:].reshape(-1, 3)]
        pvt_rs_float = pvt_rs_float + [[rs_float, q_float]]

    if table >= len(pvt_rs_float):
        raise ValueError(
            f"selected PVT table number {table} must be less than"
            " number of tables found {}".format(len(pvt_rs_float))
        )
    pvt = pvt_rs_float[table]
    for rs, bo in zip(pvt[0], pvt[1]):
        self.set_fvf(bo[:, 1], rs, bo[:, 0])

    if pres_units == "bar":
        self.bo["pres"] = self.bo.pres / 10

    return {}


class Gas(Fluid):
    """Gas fluid class.

    Attributes:
        name (str): name of the fluid
    """

    @mutually_exclusive("gas_sg", "gas_density")
    def __init__(self, name=None, gas_sg=None, gas_density=None):
        """
        gas_sg and gas_density are mutually exclusive inputs.

        Args:
            name (str, optional): Name of fluid. Defaults to None.
            gas_sg (float, optional): Gas specific gravity.
            gas_density (float, optional): Gas density (g/cc)
        """
        super().__init__(name=name)
        if gas_density is not None:
            self.set_gas_density(gas_density)
        if gas_sg is not None:
            self.set_gas_sg(gas_sg)

    def set_gas_sg(self, gas_sg):
        """

        Args:
            gas_sg ([type]): [description]
        """
        self.gas_sg = gas_sg
        self.gas_density = self.gas_sg * bw92.AIR_RHO

    def set_gas_density(self, gas_density):
        """[summary]

        Args:
            gas_density ([type]): [description]
        """
        self.gas_density = gas_density
        self.gas_sg = self.gas_density / bw92.AIR_RHO

    def density(self, temp, pres):
        """This function overides the density function in etlpy.pem.Fluid"""
        return bw92.gas_oga_density(temp, pres, self.gas_sg)

    def modulus(self, temp, pres):
        """This function overides the modulus function in etlpy.pem.Fluid"""
        return bw92.gas_adiabatic_bulkmod(temp, pres, self.gas_sg)

    def get_summary(self):
        summary = super().get_summary()
        summary.update({"density": self.gas_density, "sg": self.gas_sg})
        return summary


class GasPVDG(Gas):
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

    def load_pvdg(self, filepath, units="METRIC"):
        """

        Args:
            filepath ([type]): [description]
            units (str, optional): [description]. Defaults to 'METRIC'.
        """
        _ut = EclUnitScaler[units].value

        rawpvd = read_eclipsekw_2dtable(filepath, "PVDG")
        pvd = list()
        for rawtab in rawpvd:
            tab = dict()
            tab_ = np.array(rawtab, dtype=float).reshape(-1, 3)
            tab["pressure"] = _ut["pressure"] * tab_[:, 0]
            tab["bg"] = tab_[:, 1].copy()
            tab["visc"] = tab_[:, 2].copy()
            pvd.append(tab)
        self.pvd = pvd

        try:
            dens = read_eclipsekw_2dtable(filepath, "DENSITY")[0]
            dens = [_ut["density_kg"] * float(d) for d in dens]
        except KeyError:
            grav = read_eclipsekw_2dtable(filepath, "GRAVITY")
            raise NotImplementedError(
                "GRAVITY KW not yet implemented contact ETLP for help."
            )
            # convert to density

        self.set_gas_density(dens[2])

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

    def get_summary(self):
        return {comp: self.components[comp].get_summary() for comp in self.components}
