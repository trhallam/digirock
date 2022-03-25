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
# # Well Log Fluid Substitution

# %%
import numpy as np
import pandas as pd
import os
import pathlib
from digirock.datasets import fetch_example_data

import matplotlib.pyplot as plt

# %% [markdown]
# Fetch and setup the example data, which includes well F-4 from the Volve field.

# %%
data = fetch_example_data()

# %%
logs = pd.read_csv(data["volve_f4.csv"], index_col=0)
logs["VELP"] = 304800.0 / logs["DT"] # get to meters per second
logs["VELS"] = 304800.0 / logs["DTS"] # get to meters per second

# load some formation volume factor tables for the oil
bo_tab = pd.read_csv(data["PVT_BO.inc"], delim_whitespace=True, header=None, names=["pres", "bo"])
rs_tab = pd.read_csv(data["PVT_RS.inc"], delim_whitespace=True, header=None, names=["pres", "rs"])

# pres is in bar -> convert to MPa for digirock
bo_tab["pres"] = bo_tab["pres"] * 0.1
rs_tab["pres"] = rs_tab["pres"] * 0.1

# %%
logs.describe()

# %% [markdown]
# ## Setup the PEM
#
# We need to first build the basic blocks of our rock model including an oil/water `WoodsFluid` and Voigt-Reuss-Hill grain matrix.

# %%
from digirock import WaterBW92, OilPVT, WoodsFluid, Mineral, VRHAvg

# FLUID PROPS
temp = 110  # reservoir temperature in degC
sal = 151200  # reservoir brine salinity ppm
pres = 32 # reseroir insitu pressure MPa

volve_brine = WaterBW92(salinity=sal)
volve_oil = OilPVT(std_density=0.885, gas_sg=1.09956)
volve_oil.set_pvt(140, bo_tab["bo"].values, bo_tab["pres"].values)

fl = WoodsFluid(["sw", "so"], [volve_brine, volve_oil])

# MATRIX PROPS
volve_sand = Mineral(2.634, 31.55, 29.69, 'sand')
volve_clay = Mineral(2.78, 28.45, 7.95, 'clay')

vrh_avg = VRHAvg(["vsand", "vclay"], [volve_sand, volve_clay])

# %% [markdown]
# ## Create the Gassmann substitution Blender
#
# To perform the fluid substitution, a class which blends the props data is needed. Here we build a simple example of a `GassmannSub` class but it could be arbitrarilly more complex.

# %%
from digirock import Blend
from digirock.models import dryframe_acoustic, gassmann_fluidsub
from digirock.elastic import acoustic_bulk_moduli, acoustic_shear_moduli, acoustic_velp, acoustic_vels

class GassmannSub(Blend):
    
    _methods = ["vp", "vs", "density", "bulk_modulus", "shear_modulus"]
    
    def __init__(
        self,
        zero_porosity_model,
        fluid_model,
        sw_key="sw",
        vp_key="vp",
        vs_key="vs",
        rho_key="rho",
        poro_key="poro", 
        name=None
    ):
        elements = [zero_porosity_model, fluid_model]
        blend_keys = [sw_key, vp_key, vs_key, rho_key, poro_key]
        super().__init__(blend_keys, elements, self._methods, name=name)
        
    def _get_elastic_keys(self):
        return self.blend_keys[1:4]
    
    def dry_bulk_modulus(self, props, **kwargs):
        kfl = self.elements[1].bulk_modulus(props, **kwargs)
        k0 = self.elements[0].bulk_modulus(props, **kwargs)
        vpk, vsk, rk = self._get_elastic_keys()
        porok = self.blend_keys[4]
        ksat = acoustic_bulk_moduli(props[vpk], props[vsk], props[rk])
        kdry = dryframe_acoustic(ksat, kfl, k0, props[porok])
        return kdry
    
    def bulk_modulus(self, props, props2, **kwargs):
        kdry = self.dry_bulk_modulus(props)
        kfl = self.elements[1].bulk_modulus(props2, **kwargs)
        k0 = self.elements[0].bulk_modulus(props2, **kwargs)
        porok = self.blend_keys[4]
        return gassmann_fluidsub(kdry, kfl, k0, props2[porok])
    
    def shear_modulus(self, props, **kwargs):
        return acoustic_shear_moduli(props[self.blend_keys[2]], props[self.blend_keys[3]])
    
    def density(self, props, props2, **kwargs):
        porok = self.blend_keys[4]
        vpk, vsk, rk = self._get_elastic_keys()
        kfl1 = self.elements[1].density(props, **kwargs)
        den_min = props[rk] - props[porok] * kfl1
        kfl2 = self.elements[1].density(props2, **kwargs)
        return den_min + props2[porok] * kfl2
    
    def vp(self, props, props2, **kwargs):
        k = self.bulk_modulus(props, props2, **kwargs)
        mu = self.shear_modulus(props, **kwargs)
        den = self.density(props, props2, **kwargs)
        return acoustic_velp(k, mu, den)

    def vs(self, props, props2, **kwargs):
        mu = self.shear_modulus(props, **kwargs)
        den = self.density(props, props2, **kwargs)
        return acoustic_vels(mu, den)
        
    
test_props = {"sw":0.5, "pres":pres, "temp":temp, "vclay":0.5, "poro":0.2, "vp":3500, "vs":2500, "rho":2.2}
test_props2 = {"sw":1.0, "pres":pres, "temp":temp, "vclay":0.5, "poro":0.2}

gassub = GassmannSub(vrh_avg, fl)
print("Dry Bulk Mod: ", gassub.dry_bulk_modulus(test_props))
print("Subbed Bulk Mod: ", gassub.bulk_modulus(test_props, test_props2))
print("Subbed Shear Mod: ", gassub.shear_modulus(test_props))
print("Subbed Density: ", gassub.density(test_props, test_props2))
print("Subbed VP: ", gassub.vp(test_props, test_props2))
print("Subbed VS: ", gassub.vs(test_props, test_props2))

# %%
gassub.tree

# %% [markdown]
# ## Performing fluid substitution on the logs
#
# We simply need to define the insitu state `props1` and the state at which we wish to calculate the new logs `props2`. And these can be passed to `gassub`, our model class, which handles the details of the fluid substitution.

# %%
fig, axs = plt.subplots(ncols=5, figsize=(10,6), sharey=True)

sub = logs[np.logical_and(logs.index > 3250, logs.index < 3420)]

axs[0].plot(sub["VSH"], -sub.index, color="green")
axs[1].plot(sub["SW_SURVEY1"], -sub.index)
axs[1].plot(sub["SW_SURVEY2"], -sub.index)
axs[2].plot(sub["VELP"], -sub.index)
axs[3].plot(sub["VELS"], -sub.index)
axs[4].plot(sub["DENS"], -sub.index)

props1 = {
    "pres":pres, 
    "temp":temp, 
    "poro":sub["PORO"].values, 
    "vclay":sub["VSH"].values,
    "vp":sub["VELP"].values,
    "vs":sub["VELS"].values,
    "rho":sub["DENS"].values,
    "sw":sub["SW_SURVEY1"].values
}
props2 = props1.copy()
props2["sw"] = sub["SW_SURVEY2"].values

vp2 = gassub.vp(props1, props2)
vs2 = gassub.vs(props1, props2)
rho2 = gassub.density(props1, props2)

axs[2].plot(vp2, -sub.index)
axs[3].plot(vs2, -sub.index)
axs[4].plot(rho2, -sub.index)

axs[0].set_title("VSH")
axs[1].set_title("SW1, SW2")
axs[2].set_title("VP1, VP2")
axs[3].set_title("VS1, VS2")
axs[4].set_title("RHOB1, RHOB2")
