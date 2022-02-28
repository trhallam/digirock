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

# %% [markdown]
# # Switches

# %%
from digirock import Switch, load_pvtw
from digirock.datasets import fetch_example_data

example_data = fetch_example_data()
    
pvtos = load_pvtw(example_data["COMPLEX_PVT.inc"])
sw = Switch("pvt", list(pvtos.values()), ["density", "velocity", "bulk_modulus"], name='PVT_Water')

# %%
sw.all_keys()

# %%
props = dict(temp=110, pres=50, pvt=[0, 7, 6])
print("Bulk Modulus in 3 PVT Zones (switches)\n", sw.bulk_modulus(props))
print("Density in 3 PVT Zones (switches)\n", sw.density(props))
print("Velocity in 3 PVT Zones (switches)\n", sw.velocity(props))

# %%
