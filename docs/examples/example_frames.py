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
from digirock import VRHAvg, FixedPoroAdjust, NurCriticalPoro
from digirock.models import _mod
from digirock.elastic import acoustic_velp, acoustic_vels
from digirock.utils._decorators import check_props
import numpy as np

       

rf = VRHAvg(["VSAND", "VCLAY"], [Quartz, Anhydrite])
pa = FixedPoroAdjust(["poro"], rf)
pn = NurCriticalPoro(["poro"], rf, 0.39)
rf.shear_modulus({"VCLAY":np.linspace(0, 1, 11)})
# Quartz.trace_tree({}, "bulk_modulus", ])

# %%
pn.tree

# %%
pn.trace_tree({"VSAND":0.5, "poro":np.r_[0.2, 0.3]}, ["bulk_modulus", "vp", "density", "vs", "shear_modulus"])

# %%
rf.trace_tree({"VSAND":0.5}, ["density", "vp", "vs"])

# %%
import matplotlib.pyplot as plt

plt.plot(np.linspace(0, 1, 11), rf._vrh_avg_moduli({"VCLAY":np.linspace(0, 1, 11)}))


# %%
vsh1 = np.linspace(0, 0.5, 11)
vsh2 = np.linspace(0.5, 0.1, 11)

plt.plot(vsh1, voigt_upper_bound(10, vsh1, 20, vsh2, 30))
plt.plot(vsh1, reuss_lower_bound(10, vsh1, 20, vsh2, 30))
plt.plot(vsh1, vrh_avg(10, vsh1, 20, vsh2, 30))

# %%
