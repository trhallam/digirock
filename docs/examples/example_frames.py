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

# %% [markdown]
# # Frames
#
# `RockFrame` classes are used in `digirock` to build the dry rock frames. They inherit the `Blend` class.

# %%
from digirock import Blend, Element, Transform
from digirock.typing import NDArrayOrFloat
from typing import List, Dict, Type, Sequence
from digirock import Quartz, Anhydrite
from digirock.models import _mod
from digirock.elastic import acoustic_velp, acoustic_vels
from digirock.utils._decorators import check_props
from digirock import RockFrame

# %%
from digirock import VRHAvg, FixedPoroAdjust, NurCriticalPoroAdjust, LGStressModel, MacBethStressAdjust, Mineral

import numpy as np
import matplotlib.pyplot as plt

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


props = dict(
    VCLAY=0.2*(1-0.27),
    poro=0.27,
    temp=20,
    pres=10
)

print(hs.bulk_modulus(props))
print(hs.shear_modulus(props))

# %%
hs.tree

# %%
