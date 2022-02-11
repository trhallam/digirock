# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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
from digirock import Water, WaterPVTW

# Initialisation of BW92 Water requires the salinity in PPM.
wat = Water(name="water", salinity=0)

# check the summary - note the input salinity has been converted from PPM to ...
wat.get_summary()

# %% [markdown]
# Then let's check the elastic properties of this water with a mixture of constants and arrays or both.

# %%
# density
print("Density single values (g/cc):", wat.density(t, p))
print("Density 1 array values (g/cc):", wat.density(t, pres))

# arrays can be used, but they must be the same shape
print("Density 2 array values (g/cc):", wat.density(temp, pres))
print("Density 2 array values (g/cc):", wat.density(temp.reshape(2, -1), pres.reshape(2, -1)), '\n')

# velocity
print("Velocity single values (m/s):", wat.density(t, p))
print("Velocity 1 array values (m/s):", wat.density(t, pres))

# arrays can be used, but they must be the same shape
print("Velocity 2 array values (m/s):", wat.density(temp, pres))
print("Velocity 2 array values (m/s):", wat.density(temp.reshape(2, -1), pres.reshape(2, -1)), '\n')

# modulus
print("Modulus single values (GPa):", wat.density(t, p))
print("Modulus 1 array values (GPa):", wat.density(t, pres))

# arrays can be used, but they must be the same shape
print("Modulus 2 array values (GPa):", wat.density(temp, pres))
print("Modulus 2 array values (GPa):", wat.density(temp.reshape(2, -1), pres.reshape(2, -1)))


# %% [markdown]
# An inbuilt class exists for using PVTW tables from Eclipse include files. The density is then calculated using the Eclipse formula.

# %%
wat_pvtw = WaterPVTW(name="wat_pvtw", salinity=0)

# load the table directly from a text file
wat_pvtw.load_pvtw("example_data/COMPLEX_PVT.inc")

# look at the first value of the loaded table - there is one value for each of the 13 PVT zones in this example
print(wat_pvtw.pvt[0])

# we need to tell the fluid which pvt table to use either with each call to a method
print("Density single values (g/cc):", wat_pvtw.density(t, p, pvt=1))

# or we can set it permanently
wat_pvtw.set_active_pvtn(1)
print("Density array values (g/cc):", wat_pvtw.density(temp, pres), '\n')

# pvt can also be an array
print("Density all array values (g/cc):", wat_pvtw.density(temp, pres, pvt=np.arange(10)), '\n')

# the wat pvwt summary has extra information
wat_pvtw.get_summary()

# %%
from digirock import WaterECL

# %%
print("Density single values (g/cc):", a.density(t, p))

# %%
wat_pvtw.modulus(10, 1, pvt=0)

# %%
wat.modulus(25, 1)


# %%
def load_pvtw(self, filepath, units: str = "METRIC"):
    """

    Args:
        filepath ([type]): [description]
        units (str, optional): [description]. Defaults to 'METRIC'.

    Raises:
        ValueError: [description]
    """
    _ut: dict = EclUnitScaler[units].value

    rawpvt = read_eclipsekw_2dtable(filepath, "PVTW")
    self.pvt = list()
    for rawtab in rawpvt:
        tab = dict()
        for val, units, name in zip(
            rawtab,
            ["pressure", "unitless", "ipressure", "unitless", "ipressure"],
            ["ref_pres", "bw", "comp", "visc", "cvisc"],
        ):
            tab[name] = _ut[units] * float(val)
        self.pvt.append(tab)

    try:
        dens = read_eclipsekw_2dtable(filepath, "DENSITY")[0]
        dens = [_ut["density"] * float(d) for d in dens]
    except KeyError:
        grav = read_eclipsekw_2dtable(filepath, "GRAVITY")
        raise NotImplementedError(
            "GRAVITY KW not yet implemented contact via Github for help."
        )
        # convert to density

    self.density_asc = dens[1]


# %%
from scipy.optimize import root_scalar

# %%
from digirock.fluids import bw92
from digirock.utils.ecl import EclStandardConditions

# %%
EclStandardConditions.PRES.value

# %%
bw92.wat_density_brine(EclStandardConditions.TEMP.value, EclStandardConditions.PRES.value, 142366.6294070789*1E-6)

# %%
bw92.wat_salinity_brine(EclStandardConditions.TEMP.value, EclStandardConditions.PRES.value, 1.108193)


# %%
from digirock._fluid import load_pvtw

# %%
a = load_pvtw("example_data/COMPLEX_PVT.inc")["pvtw0"]
a.get_summary()

# %%
a.get_summary()["salinity"] * 1E6

# %%
a.density(p, t)

# %%
