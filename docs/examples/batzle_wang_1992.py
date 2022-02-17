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
# ### Recreate the work by Batzle and Wang 1992 to check `digirock.fluids` functionality
# Tony Hallam 2022

# %% [markdown]
# This notebook contains working code to test the functionality of bw98.py in rphys module, ensuring that the functions honor the work by B&W 1992

# %%
import numpy as np
from digirock.fluids import bw92

# %matplotlib inline
import matplotlib.pyplot as plt

# define some helper functions for plotting
def plotcomp(file, ax, extent, title=None, xlabel=None, ylabel=None):
    comp = plt.imread(file)
    ax.imshow(comp, extent=extent, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# %%
# Input parameters has defined by B&W 1992 for plotting purporses

temp_ar = np.arange(10, 350, 5)         # degC
pres_ar = np.arange(0.1, 100, 0.1)     # Mpa
sal_ar  = np.arange(0, 0.3, 0.01)
pres = np.array([0.1, 10, 25, 50])     # Mpa
gsg = [0.6, 1.2]                       # gas Gravity
or0 = [1.0, 0.88, 0.78]                # oil density re 15.6degC

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))

for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_oga_density(temp_ar, p, G), label=f'G={G}, P={p}')
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 0.6)
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Density (g/cc)')
ax[0].legend()
        
plotcomp('resources/bw_1992_gas_temp_vs_density.png', ax[1], [0, 350, 0, 0.6], 
         title='B&W 1992, Figure 2', xlabel = 'Temp (C)', ylabel = 'Density (g/cc)')
    

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 650)

for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_adiabatic_bulkmod(temp_ar, p, G)*1000, label=f'G={G}, P={p}')
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Bulk Modulus (MPa)')
ax[0].legend()

plotcomp('resources/bw_1992_gas_temp_vs_bulk.png', ax[1], [0, 350, 0, 650],
         title='B&W 1992 - Figure 3', xlabel='Temp (C)', ylabel='Bulk Modulus (MPa)')

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 0.09)

for G in gsg:
    for p in pres:
        ax[0].plot(temp_ar, bw92.gas_adiabatic_viscosity(temp_ar, p, G), label=f'G={G}, P={p}')
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Viscosity (centipoise)')
        
comp = plt.imread('resources/bw_1992_gas_temp_vs_visc.png')
ax[1].imshow(comp, extent=[0, 350, 0, 0.09], aspect='auto')
ax[1].set_title('B&W 1992 - Figure 4')
ax[1].set_xlabel('Temp (C)')
ax[1].set_ylabel('Viscosity (centipoise)')

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0.55, 1.05)

for p in pres:
    for r0 in or0:
        ax[0].plot(temp_ar, bw92.oil_density(r0, p, temp_ar))
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Oil Density (g/cc)')
        
comp = plt.imread('resources/bw_1992_oil_temp_vs_density.png')
ax[1].imshow(comp, extent=[0, 350, 0.55, 1.05], aspect='auto')
ax[1].set_title('B&W 1992 - Figure 5')
ax[1].set_xlabel('Temp (C)')
ax[1].set_ylabel('Oil Density (g/cc)')

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax[0].set_xlim(0, 70)
ax[0].set_ylim(1100, 1800)

api_ar = np.arange(0,70)
rho0_ar = 141/ (api_ar + 131.5)

ax[0].plot(api_ar, bw92.oil_velocity(rho0_ar, 15.6, 1E-4, 0.6, 50))
ax[0].set_xlabel('Oil API')
ax[0].set_ylabel('Oil Velocity (m/s)')
        
comp = plt.imread('resources/bw_1992_oil_api_vs_vp.png')
ax[1].imshow(comp, extent=[0, 70, 1.1, 1.8], aspect='auto')
ax[1].set_title('B&W 1992 - Figure 5')
ax[1].set_xlabel('API')
ax[1].set_ylabel('Velocity (km/s)')

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
ax[0].set_xlim(0, 350)
ax[0].set_ylim(0, 30)

for p in pres:
    for r0 in or0:
        oil_rho = bw92.oil_density(r0, p, temp_ar)
        oil_vp = bw92.oil_velocity(r0, p, temp_ar, 0.6, 50)
        ax[0].plot(temp_ar, bw92.bulkmod(oil_rho*10, oil_vp),label=f"{r0} {p}MPa")
ax[0].set_xlabel('Temp (C)')
ax[0].set_ylabel('Oil Bulk Modlus (MPa)')
ax[0].legend()#cols=2)
        
comp = plt.imread('resources/bw_1992_oil_temp_vs_bulk.png')
ax[1].imshow(comp, extent=[0, 350, 0, 3200], aspect='auto')
ax[1].set_title('B&W 1992 - Figure 5')
ax[1].set_xlabel('Temp (C)')
ax[1].set_ylabel('Oil Bulk Modlus (MPa)')

# %%
presv = [50, 100, 110]
presrho = [9.81, 49, 98.1]
presk = [0.1, 50, 100]
sal = np.array([20000, 150000, 240000])/1000000  # ppm to weigh fraction
salk = np.array([0, 150000, 300000])/1000000     # ppm to weigh fraction

tvpmesh, prvtmesh = np.meshgrid(temp_ar, presrho)
wdpmesh = bw92.wat_density_pure(tvpmesh, prvtmesh)

db1, db2, db3 = np.meshgrid(temp_ar, presrho, sal)
wdbmesh = bw92.wat_density_brine(db1, db2, db3)

tvpmesh, pvvtmesh = np.meshgrid(temp_ar, presv)
wvpmesh = bw92.wat_velocity_pure(tvpmesh, pvvtmesh)
vb1, vb2, vb3 = np.meshgrid(temp_ar, presv, sal)
wvbmesh = bw92.wat_velocity_brine(vb1, vb2, vb3)

kb1, kb2, kb3 = np.meshgrid(temp_ar, presk, salk)
kr = bw92.wat_density_brine(kb1, kb2, kb3)
kv = bw92.wat_velocity_brine(kb1, kb2, kb3)
wkbmesh = bw92.wat_bulkmod(kr, kv)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,21))

ax[0,0].set_xlim(0, 350)
ax[0,0].set_ylim(0.7, 1.3)
ax[1,0].set_xlim(0, 350)
ax[1,0].set_ylim(500, 2000)
ax[2,0].set_xlim(0, 350)
ax[2,0].set_ylim(0.5, 5.5)

for i, p in enumerate(presv):
    for j, s in enumerate(sal):
        ax[0,0].plot(temp_ar, wdbmesh[i, :, j])
        ax[1,0].plot(temp_ar, wvbmesh[i, :, j])
        ax[2,0].plot(temp_ar, wkbmesh[i, :, j])
    ax[0,0].plot(temp_ar, wdpmesh[i,:], linestyle='--', color='black')
    ax[1,0].plot(temp_ar, wvpmesh[i,:], linestyle='--', color='black')

ax[1,0].set_xlabel('Temp (C)')
ax[1,0].set_ylabel('Velocity (km/s)')

comp = plt.imread('resources/bw_1992_wat_temp_vs_density.png')
ax[0,1].imshow(comp, extent=[0, 350, 0.7, 1.3], aspect='auto')
ax[0,1].set_title('B&W 1992 - Figure 13')
ax[0,1].set_xlabel('Temp (C)')
ax[0,1].set_ylabel('Density (g/cc)')

comp = plt.imread('resources/bw_1992_pwat_temp_vs_vp.png')
ax[1,1].imshow(comp, extent=[0, 350, 0.5, 2], aspect='auto')
ax[1,1].set_title('B&W 1992 - Figure 12')
ax[1,1].set_xlabel('Temp (C)')
ax[1,1].set_ylabel('Velocity (km/s)')

comp = plt.imread('resources/bw_1992_wat_temp_vs_bulk.png')
ax[2,1].imshow(comp, extent=[0, 350, 0.5, 5.5], aspect='auto')
ax[2,1].set_title('B&W 1992 - Figure 12')
ax[2,1].set_xlabel('Temp (C)')
ax[2,1].set_ylabel('Velocity (km/s)')


# %%
print(bw92.mixed_density(10, 0.5, 5, 0.2, 2))
print(bw92.mixed_density([10, 10], [0.7, 0.5], [5, 5]))
print(bw92.mixed_bulkmod([50, 40], [0.9, 0.8], [20, 30]))

# %%
rho0 = 0.6
g = 0.6
Rg = np.linspace(0,200,50)
T = 110

bw_b0 = bw92.oil_fvf(rho0, g, Rg, T)

# %%
plt.plot(Rg, bw_b0)
