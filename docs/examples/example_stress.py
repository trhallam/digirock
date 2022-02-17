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
# # Stress Models
#
# Stress models describe the stress fields that we apply to rocks to calculate insitu conditions. The most common one required for Petroleum modelling is the vertical $S_v$ and effective $S_e$ stress.
#
# All stress models have three basic methods:
#  - `get_summary` which returns details of the stress model
#  - `vertical_stress` which returns the vertical stress of the model based upon some input properties, e.g. depth.
#  - `effective_stress` which returns the effective pressure, which is usually the vertical stress minus the formation pressure.
#  
# Additional properties or keyword arguments to each method can be added as differentiators when models become more complex.
#
# There are two basic types of stress models but users can define their own to fit the current API. The two basic types are functional models `FStressModel` and linear gradient models `LGStressModel`.

# %%
from digirock import FStressModel, LGStressModel
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# Let's start with a linear gradient model. A construction, it takes two arguments; the stress gradient in MPa/m `grad` and the reference pressure (MPa) `ref_pres` at the reference depth (mTVDSS) `ref_depth`. Usually `ref_depth` is the surface so we set this to 0 by default.

# %%
lg_stress = LGStressModel(0.01, 0.1, name="MyLGStressModel")

# %% [markdown]
# Like other `digirock` classes you can print a dictionary summary or a tree.

# %%
print(lg_stress.get_summary(), '\n')
lg_stress.tree

# %% [markdown]
# We can then calculate the vertical stress and effective stress at any depth for a given formation pressure.

# %%
depth = np.arange(1500, 3100, 100)
plt.plot(lg_stress.vertical_stress({"depth":depth}), depth, label="Sv")
plt.plot(lg_stress.effective_stress({"depth":depth, "pres":10.0}), depth, label="Se")
plt.xlabel("Stress (MPa)")
plt.ylabel("Depth (mTVDSS)")
plt.legend()
plt.gca().invert_yaxis()

# %% [markdown]
# We can also define custom functions that return more complex depth trends or even 3D fields for Stress.

# %%
from scipy.interpolate import PchipInterpolator

def custom_sv():
    # PchipInterpolator takes x and y and draws straight lines between them
    # we define the Interpolater outside the function so it isn't created each time we run func
    f = PchipInterpolator([0, 1500, 2200, 3000], [0, 10, 20, 25], extrapolate=True)
    def func(props, **kwargs):
        return f(props["depth"])
    # adjust the function name so the meaning is clear get_summary()
    func.__name__ = "PchipI"
    return func

f_stress = FStressModel(custom_sv())

# %%
depth = np.arange(1500, 3100, 100)
plt.plot(f_stress.vertical_stress({"depth":depth}), depth, label="Sv")
plt.plot(f_stress.effective_stress({"depth":depth, "pres":10.0}), depth, label="Se")
plt.xlabel("Stress (MPa)")
plt.ylabel("Depth (mTVDSS)")
plt.legend()
plt.gca().invert_yaxis()

# %%
print(f_stress.get_summary(), '\n')
f_stress.tree
