import numpy as np
from scipy.interpolate import PchipInterpolator

from ..typing import NDArrayOrFloat
from ..utils.ecl import E100MetricConst, EclUnitScaler


def oil_fvf_table(pres, bo, p, extrap="pchip"):
    """Calculate the oil formation volume factor from a b0 or PVT table

    The PVT table can be obtained from most ECLIPSE simulation data files.
    PVT tables are assumed to be constant for approximately reservoir temperature
    (isothermal).

    Within the defined table the interpolation is linear. Outside of the table

    Args:
        pres (array-like): pressure array
        bo (array-like): oil FVF function vs pres
        p (array-like): the pressure at which to calculate the FVF
        extrap (str) : ['pchip', 'const'] extrapolate outside of the defined table
                        pchip uses PCHIP type extrapolation (honors gradient)
                        const uses the closest value to the extrapolated point

    Returns:
        (array-like): the FVF for each value in p
    """
    linear = np.interp(p, pres, bo)
    if extrap == "const":
        return linear
    elif extrap == "pchip":
        tmin = min(pres)
        tmax = max(pres)
        pchipf = PchipInterpolator(pres, bo)
        pchip = pchipf(p)
        return np.where((p > tmin) & (p < tmax), linear, pchip)
    else:
        raise ValueError(
            f"Unknown `extrap` {extrap}, expected one of `pchip`, `const`."
        )


def e100_bw(
    pres: NDArrayOrFloat,
    ref_pres: float,
    bw: float,
    comp: float,
    visc: float,
    cvisc: float,
) -> NDArrayOrFloat:
    """Eclipse 100 method for calculating Bw

    Args:
        pres: Pressure to calculate Bw at.
        ref_pres: Reference pressure of bw, should be close to in-situ pressure (MPa).
        bw: Water formation volume factor at ref_pres (frac).
        comp: Compressibility of water at ref_pres (1/MPa)
        visc: Water viscosity at ref_pres (cP)
        cvisc: Water viscosibility (1/MPa)

    Returns:
        The formation volume factor for pres
    """
    x = comp * (pres - ref_pres)
    return bw / (1 + x + x * x / 2)


def e100_oil_density(api: NDArrayOrFloat) -> NDArrayOrFloat:
    """Calculate the oil density from API using Eclipse formula.

    $$
    \\rho_{API} = \\frac{141.5}{l_g} - 131.5;
    l_g = \\fracd{\\rho_{oil}}{\\rho_{wat}}
    $$

    Args:
        api

    Returns:
        Oil density $\\rho_{oil}$ at surface conditions (g/cc)
    """
    return (
        E100MetricConst.RHO_WAT.value
        * (141.5 / (api + 131.5))
        * EclUnitScaler.METRIC.value["density"]
    )
