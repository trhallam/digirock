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

# %% [markdown]
# # Frames
#
# `RockFrame` classes are used in `digirock` to build the dry rock frames. They inherit the `Blend` class.

# %%
from digirock.utils import create_orthogonal_props
import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", size=14)
figsize = (15, 5)

# a function for repeat plotting at stages of development
def plot_elastic(element, props):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5), sharex=True)
    # Plot density
    axes[0].plot(props["VCLAY"], element.density(props))
    axes[0].set_ylabel("Density (g/cc)")
    # Plot moduli
    axes[1].plot(props["VCLAY"], element.bulk_modulus(props), label="bulk")
    axes[1].set_ylabel("Modulus (GPa)")
    axes[1].plot(props["VCLAY"], element.shear_modulus(props), label="shear")
    axes[1].legend()
    # Plot velocity
    axes[2].plot(props["VCLAY"], element.vp(props), label="vp")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].plot(props["VCLAY"], element.vs(props), label="vs")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("VCLAY (frac)")
    fig.tight_layout()


# %%
from digirock import VRHAvg, FixedPoroAdjust, NurCriticalPoroAdjust, LGStressModel, MacBethStressAdjust, Mineral

# %% [markdown]
# Create two end point minerals for a our sand clay system.

# %%
sand = Mineral(2.7, 29.9, 22.5)
clay = Mineral(2.51, 22, 9.4)

# %% [markdown]
# Create a `VRHAvg` mixing model.

# %%
# the model -> rh
rf = VRHAvg(["VSAND", "VCLAY"], [sand, clay])

# check modulus variation with volume fraction
props = dict(VCLAY=np.linspace(0, 1, 51),)
plot_elastic(rf, props)

# %% [markdown]
# Apply Nur's critical porosity adjustment to turn average bulk modulus in a porous frame modulus.

# %%
pn = NurCriticalPoroAdjust(["poro"], rf, 0.39)
props = dict(
    VCLAY=np.linspace(0, 1, 51),
    poro=0.2,
)
plot_elastic(pn, props)

# %% [markdown]
# Create a `StressModel` and apply MacBeths Stress Adjustment

# %%
grad = 1/145.038*3.28084 # 1 psi/ft converted to MPa/m
# reference stress to surface
stressm = LGStressModel(grad, 0, 0)
# Then apply the MacBeth adjustment model
sm = MacBethStressAdjust(["depth, pres_init"], pn, stressm, 0.2, 17, 0.4, 18)
props = dict(
    VCLAY=np.linspace(0, 1, 51),
    poro=0.2,
    pres_init=60, #MPa
    depth = 3500, #m
    pres=40 # 20MPa pressure decline
)
plot_elastic(sm, props)

# %%
props = dict(
    VCLAY=np.linspace(0, 1, 2),
    poro=0.2,
    pres_init=60, #MPa
    depth = 3500, #m
    pres=40 # 20MPa depressure
)
sm.trace_tree(props, ["bulk_modulus", "vp"])

# %% [markdown]
# ## Create a Hashin-Shtrikman-Walpole Model

# %%
import numpy as np
from digirock import RockFrame, Element, Mineral, WaterBW92, HSWAvg
    
sand = Mineral(2.7, 35, 45)
clay = Mineral(2.51, 75, 31)
water = WaterBW92()
# water.vp = water.velocity
# water.vs = lambda x: 0
hs = HSWAvg(["VSAND", "VCLAY", "poro"], [sand, clay, water])

# use xarray to create an orthogonal set of properties in VCLAY and POROSITY
ds, props = create_orthogonal_props(VCLAY=np.linspace(0.1, 0.25, 16), poro=np.linspace(0.05, 0.3, 20), temp=np.r_[20], pres=np.r_[10])
ds["bulk_modulus"] = (ds.dims, hs.bulk_modulus(props))
ds["shear_modulus"] = (ds.dims, hs.shear_modulus(props))

fig, axs = plt.subplots(ncols=2, figsize=figsize)
ds.sel(temp=20, pres=10).bulk_modulus.plot(ax=axs[0])
ds.sel(temp=20, pres=10).shear_modulus.plot(ax=axs[1])

# %% [markdown]
# ## Create a Cemented Sand Model

# %%
from digirock import CementedSand, Mineral
import numpy as np
import xarray as xr

sand = Mineral(2.7, 35, 45)
cement = Mineral(2.9, 55, 50)

cs = CementedSand("VSAND", sand, "VCEMENT", cement, ncontacts=9)
cs.bulk_modulus({"poro":0.2, "VCEMENT": 0.05, "ncontacts":40})

# %%
ds, props = create_orthogonal_props(
    VCEMENT=np.linspace(0, 0.15, 16), ncontacts=np.arange(10, 100, 20), poro=np.arange(0.05, 0.1, 0.31)
)

ds["bulk_modulus"] = (ds.dims, cs.bulk_modulus(props))
fig, axs = plt.subplots(ncols=3, figsize=figsize)
ds.sel(poro=0.1).bulk_modulus.plot(ax=axs[0], vmin=0, vmax=50)
ds.sel(poro=0.2).bulk_modulus.plot(ax=axs[1], vmin=0, vmax=50)
# or slice in the porosity direction
ds.sel(poro=0.2).bulk_modulus.plot(ax=axs[2], vmin=0, vmax=50)


# %%
