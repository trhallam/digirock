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
from digirock import GassmannRock, VRHAvg, Mineral, WaterBW92, OilBW92, NurCriticalPoroAdjust, WoodsFluid
from digirock.utils import create_orthogonal_props

# %%
sand = Mineral(2.7, 29.9, 22.5)
clay = Mineral(2.51, 22, 9.4)

rf = VRHAvg(["VSAND", "VCLAY"], [sand, clay])
pn = NurCriticalPoroAdjust(["poro"], rf, 0.39)

wat = WaterBW92(name="water", salinity=100000)
oil = OilBW92("norst", api=35, gas_sg=0.6)
oil.set_rst(100)

woods = WoodsFluid(["SWAT", "SOIL"], [wat, oil], name="wfl")

gr = GassmannRock(pn, woods, name="myrock")

# %%
gr.tree

# %%
gr.trace_tree({"VCLAY":0.2, "poro":0.2, "SOIL": 0.5, "temp":110, "pres":20}, ["density", "bulk_modulus", "shear_modulus", "vp", "vs"])

# %%
