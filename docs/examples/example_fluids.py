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
import os
import pathlib
from digirock.datasets import fetch_example_data

cur_dir = os.path.abspath("")
parent_path = pathlib.Path(cur_dir)
print(parent_path)

# Presure in MPa
p = 50
pres = np.linspace(10, 100, 10)
# Temperature in degC
t = 110
temp = np.linspace(80, 150, 10)

# fetch all the example data
example_data = fetch_example_data()

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
print("Density 2 array values (g/cc):", wat.density(props_ar3), "\n")

# velocity
print("Velocity single values (m/s):", wat.density(props))
print("Velocity 1 array values (m/s):", wat.density(props_ar1))

# arrays can be used, but they must be the same shape
print("Velocity 2 array values (m/s):", wat.density(props_ar2))
print("Velocity 2 array values (m/s):", wat.density(props_ar3), "\n")

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
wat_pvtw = load_pvtw(example_data["COMPLEX_PVT.inc"], salinity=0)

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
print("Density array values (g/cc):", pvtw0.density(props_ar1), "\n")
print("Bulk Modulus array values (g/cc):", pvtw0.bulk_modulus(props_ar2), "\n")

# %% [markdown]
# ## Oil Types

# %%
from digirock import DeadOil, OilBW92, OilPVT, load_pvto
import numpy as np

# %% [markdown]
#
#
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
# An oil with gas in it can be created using the `OilBW92` class. This class uses the BW92 equations to calculate elastic properties. There are many different ways to calculate the elastic properties with this class.
#
#  - Explicit values for FVF `bo` and solution gas `rs` can be passed in the `props` keyword to the BW92 elastic equations.
#  - An RS to Pressure table can be set using the `set_rst` method. This allows the solution gas to be pressure dependent. The `rs` and `bo` props are then not required. `bo` is calculated per BW92.
#  - A constatnt RS can be set using the `set_rst` method. The `rs` and `bo` props are then not required. `bo` is calculated per BW92 for constant `rs`.

# %% [markdown]
# An example with no `rs` table.

# %%
bwoil_norst = OilBW92("norst", api=35, gas_sg=0.6)
display(bwoil_norst.tree)
display(bwoil_norst.density(dict(temp=110, pres=50, rs=110, bo=1.1)))

# %% [markdown]
# An example with a constant `rs`. Note, that `bo` and `rs` are no longer required in `props`, but if we pass them they will overwrite the use of the table.

# %%
bwoil_rsc = OilBW92("norst", api=35, gas_sg=0.6)
bwoil_rsc.set_rst(120)
display(bwoil_rsc.tree)
print("using set table: ", bwoil_rsc.density(dict(temp=110, pres=50)))
print(
    "using properties: ", bwoil_rsc.density(dict(temp=110, pres=50, rs=110, bo=1.1))
)  # overwrite table rs and bw92 bo

# %% [markdown]
# Finally, we can make `rs` pressure dependent by passing a table to `set_rst`.

# %%
bwoil_rst = OilBW92("norst", api=35, gas_sg=0.6)
bwoil_rst.set_rst([80, 100, 120], pres=[10, 40, 100])
display(bwoil_rst.tree)
print("using set table: ", bwoil_rst.density(dict(temp=110, pres=50)))

# %% [markdown]
# When the `rst` has been set in some manner it is possible to query `rs` and `bo` directly. Note, `rs` only needs the pressure.

# %%
print("Bo: ", bwoil_rst.bo({"temp": 110, "pres": 50}))
print("Rs: ", bwoil_rst.rs({"pres": 50}))

# %% [markdown]
# If you have an Eclipse PVTO table you can load those oil properties using `load_pvto`.

# %%
# load the Eclipse table directly from a text file
pvtos = load_pvto(example_data["COMPLEX_PVT.inc"], api=40)

# %% [markdown]
# `pvtos` is a dictionary, one for each pvto table. The returned fluid has the `OilPVT` class. This uses the BW92 equations for elastic properties, but eclusively uses `rs` and `bo` calculated from the tables loaded into the class.

# %%
print(pvtos.keys())

# %% [markdown]
# Access the individual pvt fluids using the appropriate key.

# %%
ecloil0 = pvtos["pvto0"]
ecloil0.tree

# %% [markdown]
# DataArrays are easy to plot

# %%
ecloil0.pvt["bo_table"].plot()

# %% [markdown]
# We can get the value of bo for any `pres` and `rs` combination. Eclipse100 PVT models are not temperature dependent for `bo`.

# %%
print("Bo: ", ecloil0.bo({"pres": 50, "rs": np.array([[100, 120], [100, 120]])}))

# %% [markdown]
# Getting elastic properties

# %%
props = dict(temp=np.r_[100, 110], pres=np.r_[50, 60], rs=np.r_[100, 150])
print("Density: ", ecloil0.density(props))

props = dict(temp=110, pres=50, rs=100)
print("Velocity: ", ecloil0.velocity(props))
print("Bulk Modulus: ", ecloil0.bulk_modulus(props))

# %% [markdown]
# # Gas Types

# %%
