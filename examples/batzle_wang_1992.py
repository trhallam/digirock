# -*- coding: utf-8 -*-
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
# __Recreate the work by Batzle and Wang 1992 to check `digirock.fluids.bw92` functionality.__
#
# Tony Hallam 2022

# %% [markdown]
# This notebook contains working code to test the functionality of `bw98.py` in `fluids` module of `digirock`, ensuring that the functions honor the work by B&W 1992.
#
# _Batzle, M., and Wang, Z. [1992]. Seismic properties of pore fluids. Geophysics, 57(11), 1396–1408._
# [Available from the SEG](https://library.seg.org/doi/10.1190/1.1443207).

# %%
import numpy as np
from digirock.fluids import bw92

# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", size=14)
figsize = (15, 5)

# %%
# Input parameters has defined by B&W 1992 for plotting purporses

temp_ar = np.arange(10, 350, 5)         # degC
pres_ar = np.arange(1, 100, 0.1)       # Mpa
sal_ar  = np.arange(0, 0.3, 0.01)
pres = np.array([0.1, 10, 25, 50])     # Mpa
temps = np.array([10, 100, 200, 350])  # degC
gsg = [0.6, 1.2]                       # gas Gravity
or0 = [1.0, 0.88, 0.78]                # oil density re 15.6degC

# %% [markdown]
# ## GAS
#
# Hydrocarbon density as a function of temperature and pressure using `bw92.gas_oga_density`, BW92 Eq 10a.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_oga_density(temp_ar, p, G), label=f'G={G}, P={p}')
        
    for t in temps:
        ax[1].plot(pres_ar, bw92.gas_oga_density(t, pres_ar, G), label=f'G={G}, T={t}')
        
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 0.6)
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Density (g/cc)')
ax[0].legend()
_ = ax[0].set_title('B&W 1992, Figure 2')

ax[1].set_xlim(0, 50)
ax[1].set_ylim(0, 0.6)
ax[1].set_xlabel('Pressure (MPa)')
ax[1].set_ylabel('Density (g/cc)')
_ = ax[1].legend()


# %% [markdown]
# Gas adibatic bulk modulus using `bw92.gas_adiabatic_bulkmod`.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_adiabatic_bulkmod(temp_ar, p, G)*1000, label=f'G={G}, P={p}')
        
    for t in temps:
        ax[1].plot(pres_ar, bw92.gas_adiabatic_bulkmod(t, pres_ar, G)*1000, label=f'G={G}, T={t}')

ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 650)
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Bulk Modulus (MPa)')
ax[0].legend()
ax[0].set_title('B&W 1992 - Figure 3')

ax[1].set_xlim(0, 50)
ax[1].set_xlabel('Pressure (MPa)')
_ = ax[1].legend()

# %% [markdown]
# Gas viscosity using `bw92.gas_adiabatic_viscosity` using equations 12 and 13.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)


for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_adiabatic_viscosity(temp_ar, p, G), label=f'G={G}, P={p}')
        
    for t in temps:
        ax[1].plot(pres_ar, bw92.gas_adiabatic_viscosity(t, pres_ar, G), label=f'G={G}, T={t}')
    
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Viscosity (centipoise)')
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 0.09)
ax[0].set_title('B&W 1992 - Figure 4')

ax[1].set_xlabel('Pressure (MPa)')
ax[1].set_xlim(0, 50)
_ = ax[1].legend()

# %% [markdown]
# ## OIL
#
# Dead oil density using `bw92.oil_density`, BW92 eq19.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)


for p in pres:
    for r0 in or0:
        ax[0].plot(temp_ar, bw92.oil_density(r0, p, temp_ar), label=f'r0={r0}, P={p}')
        
    for t in temps:
        ax[1].plot(pres_ar, bw92.oil_density(r0, pres_ar, t), label=f'r0={r0}, T={t}')

ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Oil Density (g/cc)')
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0.55, 1.05)
ax[0].set_title('B&W 1992 - Figure 5')
ax[0].legend()

ax[1].set_xlabel('Pressure (MPa)')
ax[1].set_xlim(0, 50)
_ = ax[1].legend(loc=[1.1, 0])


# %% [markdown]
# Oil acoustic velocity using `bw92.oil_velocity`, BW92 eq 20a.

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5,5))

api_ar = np.arange(0,70) # oil api
rho0_ar = 141/ (api_ar + 131.5)

ax.plot(api_ar, bw92.oil_velocity(rho0_ar, 15.6, 1E-4, 0.6, 50))
ax.set_xlim(0, 70)
ax.set_ylim(1100, 1800)

ax.set_xlabel('Oil API')
ax.set_ylabel('Oil Velocity (m/s)')
ax.set_title('B&W 1992 - Figure 6')


# %% [markdown]
# Oil bulk modulus using `bw92.bulkmod`.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 30)

for r0 in or0:
    for p in pres:
        oil_rho = bw92.oil_density(r0, p, temp_ar)
        oil_vp = bw92.oil_velocity(r0, p, temp_ar, 0.6, 50)
        ax[0].plot(temp_ar, bw92.bulkmod(oil_rho*10, oil_vp),label=f"{r0} {p}MPa")
        
    for t in temps:
        oil_rho = bw92.oil_density(r0, pres_ar, t)
        oil_vp = bw92.oil_velocity(r0, pres_ar, t, 0.6, 50)
        ax[1].plot(pres_ar, bw92.bulkmod(oil_rho*10, oil_vp),label=f"{r0} {t}degC")
        
        
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Oil Bulk Modlus (MPa)')
ax[0].set_title('B&W 1992 - Figure 7')
ax[0].legend()#cols=2)

ax[1].set_xlabel('Pressure (MPa)')
ax[1].set_xlim(0, 50)
_ = ax[1].legend()

# %% [markdown]
# ## WATER
#
# Set up some parameters for plotting water.

# %%
presv = [50, 100, 110] # pressure MPa for velocity plots
presrho = [9.81, 49, 98.1] # pressure MPa for density plots
presk = [0.1, 50, 100] # pressure MPa for modulus plots
sal = np.array([20000, 150000, 240000])/1000000  # ppm to weigh fraction
salk = np.array([0, 150000, 300000])/1000000     # ppm to weigh fraction




# %% [markdown]
# Pure water sonic velocity using `bw92.wat_velocity_pure` and pure water density using `bw92.wat_density_pure`. The parameters Batzle and Wang use from Wilson for pure water velocity were only calibrated to 100degC and 100MPa. So the behaviour above that is a bit odd, even though the plot in the 1992 paper looks good.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)

presv = [50, 100, 130] # pressure MPa

tvp_mesh, pvvt_mesh = np.meshgrid(temp_ar, presv)
wvp_mesh = bw92.wat_velocity_pure(tvp_mesh, pvvt_mesh)
wdp_mesh = bw92.wat_density_pure(tvp_mesh, pvvt_mesh)

for i, p in enumerate(presv):
    ax[0].plot(temp_ar, wvp_mesh[i, :], label=f"{p}MPa")
    ax[1].plot(temp_ar, wdp_mesh[i, :], label=f"{p}MPa")
    
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Velocity (m/s)')
ax[0].set_title('B&W 1992 - Figure 12')
ax[0].legend()#cols=2)
ax[0].set_xlim(0, 350)
ax[0].set_ylim(500, 2000)

ax[1].set_xlabel('Temp (C)')
ax[1].set_ylabel('Density (g/cc)')
_ = ax[1].legend()

# %% [markdown]
# Brine sonic velocity using `bw92.wat_velocity_brine` and `bw92.wat_density_brine`. Again, odd behaviour due to the influence of the pure water function on the brine velocity.

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)

presv = [50, 100, 130] # pressure MPa

db1, db2, db3 = np.meshgrid(temp_ar, presrho, sal)
wdb_mesh = bw92.wat_density_brine(db1, db2, db3)
vb1, vb2, vb3 = np.meshgrid(temp_ar, presv, sal)
wvb_mesh = bw92.wat_velocity_brine(vb1, vb2, vb3)

for i, p in enumerate(presv):
    ax[0].plot(temp_ar, wvb_mesh[i, :], label=f"{p}MPa")
    ax[1].plot(temp_ar, wdb_mesh[i, :], label=f"{p}MPa")
    
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Velocity (m/s)')
ax[0].set_title('B&W 1992 - Figure 13')
ax[0].legend()#cols=2)
ax[0].set_xlim(0, 350)
ax[0].set_ylim(1000, 2500)

ax[1].set_xlabel('Temp (C)')
ax[1].set_ylabel('Density (g/cc)')
_ = ax[1].legend()

# %% [markdown]
# Brine bulk modulus using `bw92.wat_bulkmod`. This relies on calculating the velocity and density first.

# %% tags=[]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

kb1, kb2, kb3 = np.meshgrid(temp_ar, presk, salk)
kr = bw92.wat_density_brine(kb1, kb2, kb3)
kv = bw92.wat_velocity_brine(kb1, kb2, kb3)
wkb_mesh = bw92.wat_bulkmod(kr, kv)

for i, p in enumerate(presv):
    ax[0].plot(temp_ar, wkb_mesh[i, :], label=f"{p}MPa")
    
kb1, kb2, kb3 = np.meshgrid(pres_ar, temps, salk)
kr = bw92.wat_density_brine(kb2, kb1, kb3)
kv = bw92.wat_velocity_brine(kb2, kb1, kb3)
wkb_mesh = bw92.wat_bulkmod(kr, kv)    

for i, t in enumerate(temps):
    ax[1].plot(pres_ar, wkb_mesh[i, :], label=f"{t}degC")
    
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Bulk Modulus (GPa)')
ax[0].set_ylim(0.5, 5.5)
ax[0].set_title('B&W 1992 - Figure 14')
ax[0].legend()#cols=2)

ax[1].set_xlabel('Pressure (MPa)')
ax[1].set_ylabel('Bulk Modulus (GPa)')
_ = ax[1].legend()

# %% [markdown]
# ## Other Methods
#
# For a full list of the BW92 equations available with `digirock` see the [`digirock.fluids.bw92` api](../api/fluid_methods.html#batzle-and-wang-92).
