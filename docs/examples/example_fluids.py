# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown] tags=[]
# # Fluids
#
# All fluids have four basic methods:
#  - `get_summary` which returns details of the fluid
#  - `density` which returns the density of the fluid with minimum arguments of temperature and pressure.
#  - `velocity` which returns the acoustic velocity of the fluid with minimum arguments of temperature and pressure.
#  - `modulus` which returns the modulus of the fluid with minimum arguments of temperature and pressure.
#  
# Additional keyword arguments to each method can be added as differentiators when fluids become more complex, for example when there are PVT zones.

# %% [markdown] tags=[]
# ## Water Types
#
# Classes for Water are based upon the Batzle and Wang 92 equations or with `PVTW` can be modified to use a table for the formation volume factor (FVF).

# %%
import numpy as np
# Presure in MPa
p = 50
pres = np.linspace(10, 100, 10)
# Temperature in degC
t = 110
temp = np.linspace(80, 150, 10)

# %%
from digirock import WaterBW92, WaterECL, load_pvtw

# Initialisation of BW92 Water requires the salinity in PPM.
wat = WaterBW92(name="water", salinity=0)

# check the summary - note the input salinity has been converted from PPM to ...
print(wat.get_summary())
wat.tree

# %% [markdown]
# Then let's check the elastic properties of this water with a mixture of constants and arrays or both.

# %%
props = dict(temp=t, pres=p)
props_ar1 = dict(temp=t, pres=pres)
props_ar2 = dict(temp=temp, pres=pres)
props_ar3 = dict(temp=temp.reshape(2, -1), pres=pres.reshape(2, -1))

# density
print("Density single values (g/cc):", wat.density(props))
print("Density 1 array values (g/cc):", wat.density(props_ar1))

# arrays can be used, but they must be the same shape
print("Density 2 array values (g/cc):", wat.density(props_ar2))
print("Density 2 array values (g/cc):", wat.density(props_ar3), '\n')

# velocity
print("Velocity single values (m/s):", wat.density(props))
print("Velocity 1 array values (m/s):", wat.density(props_ar1))

# arrays can be used, but they must be the same shape
print("Velocity 2 array values (m/s):", wat.density(props_ar2))
print("Velocity 2 array values (m/s):", wat.density(props_ar3), '\n')

# modulus
print("Modulus single values (GPa):", wat.density(props))
print("Modulus 1 array values (GPa):", wat.density(props_ar1))

# arrays can be used, but they must be the same shape
print("Modulus 2 array values (GPa):", wat.density(props_ar2))
print("Modulus 2 array values (GPa):", wat.density(props_ar3))



# %% [markdown]
# An inbuilt class exists for using PVTW tables from Eclipse include files. The density is then calculated using the Eclipse formula.

# %%
# load the Eclipse table directly from a text file
wat_pvtw = load_pvtw("example_data/COMPLEX_PVT.inc", salinity=0)

# look at the first value of the loaded table - there is one value for each of the 13 PVT zones in this example
print(wat_pvtw["pvtw0"].get_summary())
wat_pvtw["pvtw0"].tree

# %% [markdown]
# Let's look at the denisty for the first table entry that was loaded for this PVTW.

# %%
pvtw0 = wat_pvtw["pvtw0"]

# we need to tell the fluid which pvt table to use either with each call to a method
print("Density single values (g/cc):", pvtw0.density(props))

# with arrays
print("Density array values (g/cc):", pvtw0.density(props_ar1), '\n')
print("Bulk Modulus array values (g/cc):", pvtw0.bulk_modulus(props_ar2), '\n')

# %% [markdown]
# ## Oil Types

# %%
from digirock import DeadOil, OilBW92, OilPVT, load_pvto
import numpy as np

### Batzle and Wang Oils

# create a basic oil
obw92 = OilBW92(api=45, gas_sg=0.7)

#Set Oil with constant solution gas (rs), which is pressure independent
obw92.set_pvt(rs=100)
print(obw92.get_summary())
print("Bo with constant RS at 110degC (pres ignored)", obw92.bo(110, 50))

#Set Oil with solution gas (rs) to pressure (pres) table
obw92.set_pvt(rs=110, pres=np.arange(9,100,0.5))
print(obw92.get_summary())
print("Bo with table RS at 110degC and pres=10Mpa", obw92.bo(110, 10))
print("Bo with table RS at 110degC and pres=10Mpa", obw92.bo(95, 12))

obw92.density(props_ar1)
obw92.velocity(props_ar1)
obw92.bulk_modulus(props_ar1)
obw92.tree

# %% [markdown]
# `DeadOil` is a class for fluids with no dissolved gas and it is initialised by either specifying an oil API or standard density.

# %%
doil_api = DeadOil(api=35)
print(doil_api.get_summary())

doil_sd = DeadOil(std_density=0.84985)
print(doil_sd.get_summary())
doil_sd.tree

# %% [markdown]
# Note that `pvt` is mentioned in the summary but isn't yet set. Default behaviour for `DeadOil` is to calculate the formation volume factor (fvf) using a constant or a presure and FVF table. 

# %%
# set a constant bo
doil_api.set_pvt(1.1)
doil_api.tree

# %%
# set a bo pres table
doil_sd.set_pvt(np.linspace(1, 1.1, 11), pres=np.linspace(0, 50, 11))
doil_sd.tree

# %% [markdown]
# If you have an Eclipse PVTO table you can load those oil properties using `load_pvto`.

# %%
pvtos = load_pvto("example_data/COMPLEX_PVT.inc", api=40)

# %% tags=[]
"""Functions that simplify loading of Eclipse fluid properties into digirock classes."""

from typing import List, Union, Dict

# pylint: disable=invalid-name,no-value-for-parameter
import numpy as np
import pandas as pd

from digirock.utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from digirock.utils.ecl import EclStandardConditions, EclUnitMap, EclUnitScaler, E100MetricConst
from digirock.utils.types import Pathlike
from digirock.utils._decorators import mutually_exclusive

from digirock.fluids import bw92, ecl
from digirock import WaterECL
from digirock import OilPVT
from digirock import GasPVT

def load_density(filepath: Pathlike, units: str = "METRIC",) -> List:
    """Load the fluid density table"""
    table = dict()
    _ut: dict = EclUnitScaler[units].value
    dens = read_eclipsekw_2dtable(filepath, "DENSITY")
    
    #water
    try:
        water = [_ut["density"] * float(d[1]) for d in dens]
    except ValueError:
        water = None
        
    #oil
    try:
        oil = [_ut["density"] * float(d[0]) for d in dens]
    except ValueError:
        oil = None
        
    #gas
    try:
        gas = [_ut["density"] * float(d[2]) for d in dens]
    except ValueError:
        gas = None
    
    return {"wat":water, "oil":oil, "gas":gas}


def load_gravity(filepath: Pathlike, units: str = "METRIC",) -> List:
    """Load the fluid gravity table"""
    table = dict()
    _ut: dict = EclUnitScaler[units].value
    dens = read_eclipsekw_2dtable(filepath, "DENSITY")
    
    #water
    try:
        wdens = _ut["density"] * E100MetricConst.RHO_WAT.value 
        water = [wdens * float(d[1]) for d in dens]
    except ValueError:
        water = None
        
    #oil
    try:
        oil = [ecl.e100_oil_density(float(d[0])) for d in dens]
    except ValueError:
        oil = None
        
    #gas
    try:
        air_sg = _ut["density"] * E100MetricConst.RHO_AIR.value 
        gas = [air_sg * float(d[2]) for d in dens]
    except ValueError:
        gas = None
    
    return {"wat":water, "oil":oil, "gas":gas}


# %% tags=[]
@mutually_exclusive("api", "std_denstiy")
def load_pvto(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvto",
    api: Union[List[float], float] = None,
    std_density: Union[List[float], float] = None,
) -> Dict[str, OilPVT]:
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
        table[tabname] = OilPVT(tabname, api=ap, std_density=sd)
        _dfs = list()
        for rs, bo in zip(subtab[0], subtab[1]):
            _tdf = pd.DataFrame(data=dict(bo=bo[:, 1], pres=_ut["pressure"] * bo[:, 0]))
            _tdf["rs"] = rs
            _dfs.append(_tdf)
        _df = pd.concat(_dfs).pivot("rs", "pres", values="bo")
        rs = _df.index.values
        pres = np.tile(_df.columns.values, rs.size).reshape(_df.shape)
        table[tabname].set_pvt(rs, _df.values, pres=pres)

    return table


pvtos = load_pvto("example_data/COMPLEX_PVT.inc", api=40)


def load_pvdg(
    filepath: Pathlike,
    units: str = "METRIC",
    prefix: str = "pvdg",
    density_asc: Union[List[float], float] = None,
) -> Dict[str, GasPVT]:
    """Load a PVDG table into multiple [`WaterECL`][digirock.WaterECL] classes.

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
        density_asc: The density of the gas/s.

    Returns:
        A dictionary of WaterECL instance for each PVT table entry.

    Raises:
        AssertionError: units is not valid
        ValueError: DENSITY kw not in file with PVTW
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

pvdgs = load_pvdg("example_data/COMPLEX_PVT.inc")

# %%
pvdgs["pvtw0"].tree

# %%
pvtos["pvto0"].pvt["bo_table"].plot()

# %%
np.tile(pres, rs.size).reshape(_df.shape).shape

# %%
_df.shape

# %%
import pandas as pd

# %%
for 

df = pd.DataFrame(index = rs)


# %%
df

# %%
from enum import Enum

class E100MetricConst(Enum):
    PRES_ATMS = 1.013  # barsa
    RHO_AIR = 1.22  # kg/m3
    RHO_WAT = 1000.0  # kg/m3
    GAS_CONST = 0.083143  # m3bars/K/kg-M
    
class EclUnitScaler(Enum):
    METRIC = dict(
        length=1,
        time=1,
        area=1,
        density=1e-3,
        density_kg=1.0,
        pressure=1e-1,
        ipressure=1 / 1e-1,
        temp_abs=1,
        temp_rel=lambda x: x,
        compress=1e2,
        viscosity=1,
        perm=1,
        volume=1,
        unitless=1,
    )


# %%
def e100_oil_density(api ):
    """Calculate the oil density from API using Eclipse formula.

    $$
    \\API = \\frac{141.5}{l_g} - 131.5;
    l_g = \\fracd{\\rho_{oil}}{\\rho_{wat}}
    $$

    Args:
        api

    Returns:
        Oil density $\\rho_{oil}$ at surface conditions (g/cc)
    """
    return E100MetricConst.RHO_WAT.value * (141.5 / (api + 131.5)) * EclUnitScaler.METRIC.value["density"]
    
e100_oil_density(4)

# %%
