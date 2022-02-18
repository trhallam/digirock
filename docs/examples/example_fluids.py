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

# %% [markdown]
# ## Blend Fluids

# %%
from digirock import WaterBW92, DeadOil, WoodsFluid
import numpy as np

# Initialisation of BW92 Water requires the salinity in PPM.
wat = WaterBW92(name="water", salinity=0)
doil_api = DeadOil(api=35)
doil_api.set_pvt(np.linspace(1, 1.1, 11), pres=np.linspace(0, 70, 11))

wf = WoodsFluid(["sw", "so"], [wat, doil_api], name="wf")

# %%
wf.tree

# %%
props = dict(temp=110, pres=50, sw=[1.0, 0.7, 0.9], so=[0.0, 0.3, 0.1])
wf.velocity(props)

# %%
