"""Classes for Oil like Fluids"""
from typing import Dict, Tuple, Union
import numpy as np
import xarray as xr

from .._exceptions import WorkflowError
from ..utils.ecl import EclStandardConditions
from ..utils._decorators import check_props, mutually_exclusive, broadcastable
from ..utils import check_broadcastable
from ..typing import NDArrayOrFloat
from ..fluids import bw92
from ..fluids import ecl as fluid_ecl

from ._fluid import Fluid


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

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update(
            {
                "api": self.api,
                "std_density": self.std_density,
            }
        )
        return summary


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
        summary = super().get_summary()
        summary.update(
            {
                "pvt": self.pvt,
            }
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
        self.rst = None

    def set_rst(
        self,
        rs: NDArrayOrFloat,
        pres: NDArrayOrFloat = None,
    ):
        """Set the RS table for Oil.

        The solution gas ratio `rs` i[] set for tables of `bo` and `pres`.

        Args:
            rs: The solution gas ratio for bo or bo table. Has shape (M, )
            pres: Pressure values (MPa) to match rs if defined. Has shape (M, ).
        """
        rs = np.atleast_1d(rs)
        pres = np.atleast_1d(pres) if pres is not None else None

        # constants for rs and bo
        if rs.size == 1:
            self.rst = {"type": "constant", "rs": rs[0], "pres": pres}
        elif len(rs.shape) > 1 and pres is None:
            raise ValueError("presure requires for list of `rs`")
        elif rs.shape != pres.shape:
            raise ValueError(
                f"`pres` {pres.shape} and `rs` {rs.shape} must have same shape"
            )
        elif rs.shape == pres.shape:
            table = xr.DataArray(data=rs, coords={"pres": pres})
            self.rst = {
                "type": "table",
                "rs": "table",
                "pres": "table",
                "rs_table": table,
            }
        else:
            raise NotImplementedError("Unknown combination of rs and bo")

    @broadcastable("temp", "pres")
    def _get_rsbo(
        self, temp: NDArrayOrFloat, pres: NDArrayOrFloat
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
        if self.rst is None:
            raise WorkflowError(
                "_get_rsbo",
                "RS/PRESSURE relationship needs to be set using `set_rst()`",
            )

        if self.rst["type"] == "constant":
            rs = self.rst["rs"]
        elif self.rst["type"] == "table":
            rs = self.rst["rs_table"].interp(pres=pres).values
        else:
            raise NotImplementedError(f"PVT of type {self.rst['type']} is unknown")

        if self.std_density is not None:
            fvf = bw92.oil_fvf(self.std_density, self.gas_sg, rs, temp)
        else:
            raise WorkflowError(
                "std_density", "Set an oil standard density or api first."
            )

        return rs, fvf

    @check_props("temp", "pres")
    def bo(self, props: Dict[str, NDArrayOrFloat]) -> NDArrayOrFloat:
        """Calculate the oil formation volume factor (bo) using BW92.

        Set the attribute `bo` using BW92 [oil_fvf][digirock.fluids.bw92.oil_fvf].

        Args:
            props: A dictionary of properties; requires `temp` (degC) and `pressure` (MPa)

        Returns:
            Formation volume factor FVF (L/L)
        """
        _, fvf = self._get_rsbo(props["temp"], props["pres"])
        return fvf

    @check_props("pres")
    def rs(self, props: Dict[str, NDArrayOrFloat]) -> NDArrayOrFloat:
        """Calculate the solution gas (rs) from pressure/rs table.

        Args:
            props: A dictionary of properties; requires `pressure` (MPa)

        Returns:
            solution gas (L/L)
        """
        # temperatue is dummy
        rs, _ = self._get_rsbo(100.0, props["pres"])
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
        summary.update({"rst": self.rst})
        return summary


class OilPVT(BaseOil):
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
            gas_sg: Dissolved gas standard gravity (required)
        """
        self.gas_sg = gas_sg
        self.pvt = None
        super().__init__(name=name, api=api, std_density=std_density)

        if self.gas_sg is None:
            raise ValueError("gas_sg cannot be None")

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

            table = xr.DataArray(
                data=bo, dims={"pres": pres}, coords={"pres": pres}
            ).expand_dims({"rs": rs})
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

    def _get_rsbo(
        self, pres: NDArrayOrFloat, rs: NDArrayOrFloat
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:

        to_shp = check_broadcastable(pres=pres, rs=rs)
        rs_l = np.broadcast_to(rs, to_shp)
        pres_l = np.broadcast_to(pres, to_shp)

        coords = {"bo": range(rs_l.size)}
        sampling_ar = dict(
            rs=xr.DataArray(rs_l.ravel(), dims="bo", coords=coords),
            pres=xr.DataArray(pres_l.ravel(), dims="bo", coords=coords),
        )

        if self.pvt["type"] == "fixed_rs":
            fvf = (
                self.pvt["bo_table"].sel(rs=rs).interp(pres=sampling_ar["pres"]).values
            )
            fvf = np.reshape(fvf, to_shp)
        elif self.pvt["type"] == "full":
            fvf = self.pvt["bo_table"].interp(**sampling_ar).values
            fvf = np.reshape(fvf, to_shp)
        else:
            raise NotImplementedError(f"PVT of type {self.pvt['type']} is unknown")

        return fvf

    def _process_rs(self, rs: Union[NDArrayOrFloat, None]) -> NDArrayOrFloat:
        # Get RS or raise error
        if rs is None and self.pvt["type"] == "fixed_rs":
            rs = self.pvt["rs"]
        elif rs is None and self.pvt["type"] == "full":
            raise WorkflowError(
                self.name, "`rs` must be passed to interpolate PVT table values for bo"
            )

        return rs

    def _process_bo_rs(
        self, props: Dict[str, NDArrayOrFloat]
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
        # If RS or BO are not supplied, calculate from the table
        rs = props.get("rs")
        fvf = props.get("bo")

        rs = self._process_rs(rs)

        if fvf is None:
            fvf_l = self._get_rsbo(props["pres"], rs)
            return fvf_l if fvf is None else fvf, rs
        else:
            return fvf, rs

    @check_props("pres")
    def bo(self, props: Dict[str, NDArrayOrFloat]) -> NDArrayOrFloat:
        """Interpolate bo from pvt table.

        Args:
            props: A dictionary of properties; requires `pressure` (MPa); `rs` (L/L) unless rs is constant.

        Returns:
            Formation volume factor FVF (L/L)
        """
        fvf, rs = self._process_bo_rs(props)
        return fvf

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
        summary.update(
            {
                "pvt": self.pvt,
                "gas_sg": self.gas_sg,
            }
        )
        return summary
