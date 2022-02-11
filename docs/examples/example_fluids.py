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
from digirock import Water, WaterECL, load_pvtw

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
# load the Eclipse table directly from a text file
wat_pvtw = load_pvtw("example_data/COMPLEX_PVT.inc", salinity=0)

# look at the first value of the loaded table - there is one value for each of the 13 PVT zones in this example
print(wat_pvtw["pvtw0"].get_summary())

# %% [markdown]
# Let's look at the denisty for the first table entry that was loaded for this PVTW.

# %%
pvtw0 = wat_pvtw["pvtw0"]

# we need to tell the fluid which pvt table to use either with each call to a method
print("Density single values (g/cc):", pvtw0.density(t, p))

# with arrays
print("Density array values (g/cc):", pvtw0.density(temp, pres), '\n')
print("Bulk Modulus array values (g/cc):", pvtw0.bulk_modulus(temp, pres), '\n')

# %% [markdown]
# ## Oil Types

# %%
from digirock import DeadOil, Oil

# %% [markdown]
# `DeadOil` is a class for fluids with no dissolved gas and it is initialised by either specifying an oil API or standard density.

# %%
doil_api = DeadOil(api=35)
print(doil_api.get_summary())

doil_sd = DeadOil(std_density=0.84985)
print(doil_sd.get_summary())

# %% [markdown]
# Note that `bo` is mentioned in the summary but isn't yet set. Default behaviour for `DeadOil` is to calculate the formation volume factor (fvf) using Batzle and Wang 92 when a bo relationship isn't specified. The `bo` is temperature specific and stored in the `bo` attribute.

# %%
# fvf is based upon temperature in degC
doil_api.calc_fvf(110)
print(doil_api.bo)

# %%
doil_api.keys()

# %%
WaterECL.keys()

# %%
