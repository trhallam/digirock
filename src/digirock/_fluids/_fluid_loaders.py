"""Functions that simplify loading of Eclipse fluid properties into digirock classes."""
from typing import List, Union, Dict
from collections.abc import Iterable

# pylint: disable=invalid-name,no-value-for-parameter
import numpy as np
import pandas as pd

from ..utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from ..utils.ecl import (
    E100MetricConst,
    EclStandardConditions,
    EclUnitMap,
    EclUnitScaler,
)
from ..typing import Pathlike
from ..utils._decorators import mutually_exclusive

from ..fluids import bw92, ecl
from ._water import WaterECL
from ._oil import OilPVT
from ._gas import GasPVT


def load_density(
    filepath: Pathlike,
    units: str = "METRIC",
) -> List:
    """Load the fluid density table"""
    table = dict()
    _ut: dict = EclUnitScaler[units].value
    dens = read_eclipsekw_2dtable(filepath, "DENSITY")

    # water
    try:
        water = [_ut["density"] * float(d[1]) for d in dens]
    except ValueError:
        water = None

    # oil
    try:
        oil = [_ut["density"] * float(d[0]) for d in dens]
    except ValueError:
        oil = None

    # gas
    try:
        gas = [_ut["density"] * float(d[2]) for d in dens]
    except ValueError:
        gas = None

    return {"wat": water, "oil": oil, "gas": gas}


def load_gravity(
    filepath: Pathlike,
    units: str = "METRIC",
) -> List:
    """Load the fluid gravity table"""
    table = dict()
    _ut: dict = EclUnitScaler[units].value
    dens = read_eclipsekw_2dtable(filepath, "GRAVITY")

    # water
    try:
        wdens = _ut["density"] * E100MetricConst.RHO_WAT.value
        water = [wdens * float(d[1]) for d in dens]
    except ValueError:
        water = None

    # oil
    try:
        oil = [ecl.e100_oil_density(float(d[0])) for d in dens]
    except ValueError:
        oil = None

    # gas
    try:
        air_sg = _ut["density"] * E100MetricConst.RHO_AIR.value
        gas = [air_sg * float(d[2]) for d in dens]
    except ValueError:
        gas = None

    return {"wat": water, "oil": oil, "gas": gas}


def load_pvtw(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvtw",
    salinity: List[int] = None,
) -> Dict[str, WaterECL]:
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
        dens = load_density(filepath, units=units)["wat"]
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
        # grav = load_gravity(filepath, units=units)
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


@mutually_exclusive("api", "std_denstiy")
def load_pvto(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvto",
    api: Union[List[float], float] = None,
    std_density: Union[List[float], float] = None,
    gas_sg: Union[List[float], float] = None,
) -> Dict[str, OilPVT]:
    """Load a PVTO table into multiple [`OilPVT`][digirock.OilPVT] classes.

    PVTO tables have the form (where units may differ):

    ```
    PVTO
    ------------------------------------------------------------
    --SOLUTION  PRESSURE  OIL FVF      OIL
    -- GOR Rs      Po       Bo      VISCOSITY
    -- Sm3/Sm3    bara    rm3/Sm3      cP
    ------------------------------------------------------------
    --
    PVTO
    -- REGION 1
        27.4     50.0    1.17014      1.224
                100.0    1.16296      1.310
                150.0    1.15664      1.393 /
        54.3    100.0    1.24802      0.953
                150.0    1.23902      1.027
                200.0    1.23114      1.099 /
    /
    -- REGION 2
        27.4     50.0    1.17014      1.224
                100.0    1.16296      1.310
                150.0    1.15664      1.393 /
        54.3    100.0    1.24802      0.953
                150.0    1.23902      1.027
                200.0    1.23114      1.099 /
    /
    ```

    `api` and `std_density` cannot be specified together. If both are `None` then the DENSITY keyword must be present in the same input file to get the oil density.

    Args:
        filepath: Filepath or str to text file containing PVTW keyword tables
        units: The Eclipse units of the PVTW table, one of ['METRIC', 'PVTM', 'FIELD', 'LAB']. Defaults to 'METRIC'
        prefix: The prefix to apply to the name of each loaded fluid
        api: A single oil API or list of API values for each PVT Region in the file
        std_density: A single standard density in g/cc at 15.6degC or a list of values for each PVT Region in the file
        gas_sg: A single standard gravity for the disovled gas or a list of values for each PVT Region in the file

    Returns:
        A dictionary of Oil instance for each PVT table entry.

    Raises:
        AssertionError: units is not valid
    """
    assert units in EclUnitMap.__members__
    _ut: dict = EclUnitScaler[units].value

    try:
        ecl_dens_tab = load_density(filepath, units=units)
    except KeyError:
        ecl_dens_tab = None

    # here but not implemented yet
    try:
        ecl_grav_tab = load_gravity(filepath, units=units)
    except KeyError:
        ecl_grav_tab = None

    if api is None and std_density is None and ecl_dens_tab is not None:
        std_density = list(ecl_dens_tab["oil"])

    if gas_sg is None and ecl_dens_tab is not None:
        gas_sg = [
            1000 * rho_g / E100MetricConst.RHO_AIR.value
            for rho_g in ecl_dens_tab["gas"]
        ]
    else:
        raise ValueError(
            "Input file does not contain DENSITY table, `gas_sg` required."
        )

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

    def _bc_helper(inv, n):
        if inv is None:
            return [None] * n
        elif isinstance(inv, Iterable):
            assert len(inv) == n
            return inv
        else:
            return [inv] * n

    # check api/std_density against ntab
    ntab = len(pvt_rs_float)
    api_iter = _bc_helper(api, ntab)
    sd_iter = _bc_helper(std_density, ntab)
    sg_iter = _bc_helper(gas_sg, ntab)

    table = dict()
    for i, (subtab, ap, sd, sg) in enumerate(
        zip(pvt_rs_float, api_iter, sd_iter, sg_iter)
    ):
        tabname = f"{prefix}{i}"
        table[tabname] = OilPVT(tabname, api=ap, std_density=sd, gas_sg=sg)
        _tdf = pd.concat(
            [
                pd.DataFrame(
                    data=dict(bo=bo[:, 1], pres=_ut["pressure"] * bo[:, 0], rs=rs)
                )
                for rs, bo in zip(subtab[0], subtab[1])
            ]
        )
        _tdf = _tdf.pivot("pres", "rs", values="bo")
        table[tabname].set_pvt(
            _tdf.columns.values,
            _tdf.values.T,
            pres=np.broadcast_to(_tdf.index.values, _tdf.values.T.shape),
        )

    return table


def load_pvdg(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvdg",
    density_asc: Union[List[float], float] = None,
) -> Dict[str, GasPVT]:
    """Load a PVDG table into multiple [`GasPVT`][digirock.GasPVT] classes.

    PVDG tables have the form (where units may differ):

    ```
    PVDG
    ------------------------------------------------------------
    --PRESSURE  VAPORIZED  GAS FVF      GAS
    --   Pg      OGR  Rv     Bg      VISCOSITY
    --  bara     Sm3/Sm3   rm3/Sm3      cP
    ------------------------------------------------------------
    -- PVT region 1
    50.0         0.024734     0.014335
    100.0        0.011846     0.015991
    150.0        0.007763     0.018581 /
    -- PVT region 2
    50.0         0.024734     0.014335
    100.0        0.011846     0.015991
    150.0        0.007763     0.018581 /
    ```


    Args:
        filepath: Filepath or str to text file containing PVTW keyword tables.
        units: The Eclipse units of the PVTW table, one of ['METRIC', 'PVTM', 'FIELD', 'LAB']. Defaults to 'METRIC'.
        prefix: The prefix to apply to the name of each loaded fluid.
        density_asc: The density of the gas/s, required if DENSITY KW is missing from file.

    Returns:
        A dictionary of GasPVT instance for each PVT table entry.

    Raises:
        AssertionError: units is not valid
        ValueError: DENSITY kw not in file with PVDG
    """
    assert units in EclUnitMap.__members__
    _ut = EclUnitScaler[units].value

    rawpvd = read_eclipsekw_2dtable(filepath, "PVDG")
    pvd = list()
    for rawtab in rawpvd:
        tab = dict()
        tab_ = np.array(rawtab, dtype=float).reshape(-1, 3)
        tab["pres"] = _ut["pressure"] * tab_[:, 0]
        tab["bg"] = tab_[:, 1].copy()
        tab["visc"] = tab_[:, 2].copy()
        pvd.append(tab)

    if density_asc is None:
        try:
            density_asc_ar = np.array(load_density(filepath, units=units)["gas"])
        except KeyError:
            # grav = load_gravity(filepath, units=units)
            raise NotImplementedError(
                "GRAVITY KW not yet implemented contact via Github for help."
            )
    elif np.atleast_1d(density_asc).size == 1:
        density_asc_ar = np.r_[density_asc].repeat(len(pvd))
    else:
        density_asc_ar = np.array(density_asc).flatten()
        # convert to density

    assert density_asc_ar.size == len(pvd)

    table = dict()
    for i, (subtab, sd) in enumerate(zip(pvd, density_asc_ar)):
        tabname = f"{prefix}{i}"
        table[tabname] = GasPVT(sd, name=tabname)
        table[tabname].set_pvt(subtab["pres"], subtab["bg"])
    return table
