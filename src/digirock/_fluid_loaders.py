"""Functions that simplify loading of Eclipse fluid properties into digirock classes."""

from typing import List, Union

# pylint: disable=invalid-name,no-value-for-parameter
import numpy as np

from .utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from .utils.ecl import EclStandardConditions, EclUnitMap, EclUnitScaler
from .utils.types import Pathlike
from .utils._decorators import mutually_exclusive

from .fluids import bw92
from ._fluid import WaterECL, OilPVT, GasECL


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


@mutually_exclusive("api", "std_denstiy")
def load_pvto(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvto",
    api: Union[List[float], float] = None,
    std_density: Union[List[float], float] = None,
) -> dict:
    """Load a PVTO table into multiple [`Oil`][digirock.Oil] classes.

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

    `api` and `std_density` cannot be specified together.

    Args:
        filepath: Filepath or str to text file containing PVTW keyword tables.
        units: The Eclipse units of the PVTW table, one of ['METRIC', 'PVTM', 'FIELD', 'LAB']. Defaults to 'METRIC'.
        prefix: The prefix to apply to the name of each loaded fluid.
        api: A single oil API or list of API values for each PVT Region in the file.
        std_density: A single standard density in g/cc at 15.6degC or a list of values for each PVT Region in the file.

    Returns:
        A dictionary of Oil instance for each PVT table entry.

    Raises:
        AssertionError: units is not valid
    """
    assert units in EclUnitMap.__members__
    _ut: dict = EclUnitScaler[units].value
    if api is None and std_density is None:
        raise ValueError("One of api or std_density must be given")

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

    # check api/std_density against ntab
    ntab = len(pvt_rs_float)
    if api is None:
        api_iter = [None] * ntab
    elif isinstance(api, list):
        assert len(api) == ntab
        api_iter = api
    else:
        api_iter = [api] * ntab

    if std_density is None:
        sd_iter = [None] * ntab
    elif isinstance(api, list):
        assert len(std_density) == ntab
        sd_iter = std_density
    else:
        sd_iter = [std_density] * ntab

    table = dict()
    for i, (subtab, ap, sd) in enumerate(zip(pvt_rs_float, api_iter, sd_iter)):
        tabname = f"{prefix}{i}"
        table[tabname] = Oil(tabname, api=ap, std_density=sd)
        for rs, bo in zip(subtab[0], subtab[1]):
            table[tabname].set_fvf(bo[:, 1], rs, pres=_ut["pressure"] * bo[:, 0])

    return table


def load_pvdg(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvtw",
) -> dict:
    """Load a PVTG table into multiple [`WaterECL`][digirock.WaterECL] classes.

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
