"""Function for calculating fluid properties in a reservoir

| Symbols  |   Definitions             |    Units          |        Function/s      |
|----------|---------------------------|-------------------|------------------------|
rhow       |  Density of brine         |   g/cm3           |
rhog       |  Density of gas           |   g/cm3           |
rhoo       |  Density of oil           |   g/cm3           |
rho0       |  Reference Density of Oil |   g/cm3           |
kwat       |  Bulk modulus of water    |   GPa             |
kgas       |  Bulk modulus of gas      |   GPa             |
koil       |  Bulk modulus of oil      |   GPa             |
g          |  Specific gravity of gas  |   API             |
p          |  In-situ pressure         |   MPa             |
t          |  In-situ temperature      |   degC            |
sal        |  Salinity                 |   Wt.fraction     |
sw         |  Water Saturation         |   fraction        |
rg         |  Gas-to-Oil ratio (GOR)   |   Litre/litre     |
gas_R      |  Gas constant             |   m3 Pa K-1 mol-1 |

Notes:
    - The gas specific gravity G is the ratio of the gas density to air density
    at 15.6 degC and at atmospheric pressure. Typically gases have G values
    from 0.56 (Methane) to 1.8

    - Salinity has units that can be converted as such
    35 g dissolved salt / kg sea water = 35 ppt = 35 o/oo = 3.5% = 35000 ppm = 35000 mg/l

| Constant | Variable Name | Units | Value |
|----------|---------------|-------|-------|
| Gas Constant | `GAS_R`   | (J K-1 mol-1) (m3 Pa K-1 mol-1) | 8.31441 |
| Atmospheric Pressures | `STD_ATM_PRESS` | Pa | 101325 |
| Molecular Weight of Air | `AIR_M_WEIGHT` | g/mol | 28.967 |
| Density of air at STD | `AIR_RHO` | g/m3 | AIR_M_WEIGHT * STD_ATM_PRESS / ((15.6 + 273.15) * GAS_R) |

These functions are based upon the work by:

   1. Batzle and Wang, 1992, Seismic Properties of Pore Fluids
   2. Kumar, D, 2006, A Tutorial on Gassmann Fluid Substitution: Formulation,
        Algorithm and Matlab Code

"""
import warnings

import numba
import numpy as np

from scipy.optimize import root_scalar

from digirock.utils.types import NDArrayOrFloat

from ..utils import safe_divide

#  define constants
GAS_R = 8.31441  #  gas constant (J K-1 mol-1) (m3 Pa K-1 mol-1)
STD_ATM_PRESS = 101325  #  standard atmospheric pressure (Pa)
AIR_M_WEIGHT = 28.967  #  molecular weight of air (PetroWiki) (g/mol)
AIR_RHO = (
    AIR_M_WEIGHT * STD_ATM_PRESS / ((15.6 + 273.15) * GAS_R)
)  # density of air at standard cond (g/m3)
AIR_RHO = AIR_RHO / 1000  # density of air at standard cond (kg/m3)

WATC = np.array(
    [  # coefficients for water velocity
        [1402.85, 1.524, 3.437e-3, -1.197e-5],
        [4.871, -0.0111, 1.739e-4, -1.628e-6],
        [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
        [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
        [-2.197e-7, 7.987e-10, 5.23e-11, -4.614e-13],
    ]
)


def gas_vmol(t, p):
    """Calculates molar volume for and ideal gas

    B&W 1992 Eq 1

    Args:
        t (array-like): Temperature of the gas in degC
        p (array-like): The pressure of the gas in Pa

    Returns:
        (array-like): the molar volume of the gas for ta and p

    """
    return GAS_R * (t + 273.15) / p


def gas_density(m, t, p):
    """Calculates the density for and ideal gas

    B&W 1992 Eq 2

    Args:
        m (array-like): Molecular weight of the gas.
        t (array-like): Temperature of the gas in degC
        p (array-like): Pressure of the gas in MPa

    Returns:
        (array-like): the molar weight of the gas for ta and p

    """
    return np.divide(np.multiply(m, p), (t + 273.15) * GAS_R)


def gas_isotherm_comp(v1, v2, p1, p2):
    """Calculates the isothermal compressibility of an ideal gas

    Two points of corresponding molecular volume and pressure must be known to calculate Bt
    for a given constant temperature.
    B&W 1992 Eq 3

    Args:
        v1 (array-like): The first molecular volume of the gas
        v2 (array-like): The second molecular volume of the gas
        p1 (array-like): The first pressure of the gas in Pa
        p2 (array-like): The second pressure of the gas in Pa

    Returns:
        Bt (array-like): the isothermal compressibility
    """
    return -1 * (v2 - v1) / (p2 - p1) / v1


def gas_isotherm_vp(m, t):
    """Calculates the isothermal velocity of a compressional wave for an ideal gas

    B&W 1992 Eq 4

    Args:
        m (array-like): Molecular weight of gas
        t (array-like): Temperature of gas in degC

    Returns:
        vp (array-like): Isothermal compressional velocity for gas
    """
    return np.sqrt(GAS_R * (t + 273.15) / m)


def gas_pseudocrit_pres(g):
    """Calculates the gas pseudo critical pressure value

    B&W 1992 Eq 9a

    Args:
        g (array-like): The gas specific gravity

    Returns:
        (array-like): The pseudo critical pressure

    """
    return 4.892 - 0.4048 * g


def gas_pseudored_pres(p, g):
    """Calculates the gas pseudo reduced pressure value

    B&W 1992 Eq 9a

    Args:
        p (array-like): The gas pressure in MPascals
        g (array-like): The gas specific gravity

    Returns:
         (array-like): the pseudo reduced or normalised pseudo critical pressure value

    """
    return p / gas_pseudocrit_pres(g)


def gas_pseudocrit_temp(g):
    """Calculates the gas pseudo critical temperature value

    B&W 1992 Eq 9b

    Args:
        g (array-like): The gas specific gravity

    Returns:
         (array-like): The pseudo critical temperature

    """
    return 94.72 + 170.75 * g


def gas_pseudored_temp(t, g):
    """Calculates the gas pseudo reduced pressure value

    B&W 1992 Eq 9b

    Args:
        t (array-like): The gas absolute temperature in degK
        g (array-like): The gas specific gravity

    Returns:
        (array-like): the pseudo reduced or normalised pseudo critical temperature value

    """
    return (t + 273.15) / gas_pseudocrit_temp(g)


def gas_oga_density(t, p, g):
    """Calculates the approximate density of gas appropriate to the oil and gas industry

    Suitable for temperatures and pressures typically encountered in the oil and gas
    industry. This approximation is adequate as long as ppr and tpr are not both within
    about 0.1 of unity.
    B&W 1992 Eq 10a

    Args:
        t (array-like): The gas absolute temperature in degK
        p (array-like): The gas pressure in MPa
        g (array-like): The gas specific gravity

    Returns:
        (array-like): the oil and gas appropriate approximate density in g/cc

    """
    ppr = gas_pseudored_pres(p, g)
    tpr = gas_pseudored_temp(t, g)

    if np.any(np.logical_and(np.abs(ppr - 1) < 0.1, np.abs(tpr - 1) < 0.1)):
        warnings.warn(
            "Values for ppr and trp are both within 0.1 of unity, the gas \
            density assumptions will not be valid."
        )

    a = 0.03 + 0.00527 * (3.5 - tpr) ** 3
    b = 0.642 * tpr - 0.007 * tpr**4 - 0.52
    c = 0.109 * (3.85 - tpr) ** 2
    d = np.exp(-1 * (0.45 + 8 * (0.56 - 1 / tpr) ** 2) * (ppr**1.2) / tpr)

    E = c * d
    Z = a * ppr + b + E

    return 28.8 * np.divide(np.multiply(g, p), GAS_R * np.multiply(Z, (t + 273.15)))


def gas_adiabatic_bulkmod(t, p, g):
    """Calculates the approximate adiabatic (constant pressure) bulk modulus of a gas

    Args:
        t (array-like): Gas temperature in degC
        p (array-like): The gas pressure in MPa
        g (array-like): The gas specific gravity

    Returns:
        (array-like): approximate gas bulk modulus in GPa

    """
    ppr = gas_pseudored_pres(p, g)
    tpr = gas_pseudored_temp(t, g)
    a = 0.03 + 0.00527 * (3.5 - tpr) ** 3
    b = 0.642 * tpr - 0.007 * tpr**4 - 0.52
    c = 0.109 * (3.85 - tpr) ** 2
    d = np.exp(-1 * (0.45 + 8 * (0.56 - 1 / tpr) ** 2) * (ppr**1.2) / tpr)
    gamma = (
        0.85
        + 5.6 / (ppr + 2)
        + 27.1 / np.square(ppr + 3.5)
        - 8.7 * np.exp(-0.65 * (ppr + 1))
    )
    m = 1.2 * (-1 * (0.45 + 8 * (0.56 - 1 / tpr) ** 2) * (ppr**0.2) / tpr)
    f = c * d * m + a
    Z = a * ppr + b + c * d
    return p * gamma * 1e-3 / (1 - ppr * f / Z)


def gas_adiabatic_viscosity(t, p, g):
    """Calculates the approximate adiabatic (constant pressure) viscoisty of a gas

    Args:
        t (float): The gas temperature in degC
        p (float): The gas pressure in MPascals
        g (float): The gas specific gravity

    Returns:
        (float): approximate gas bulk modulus in centipoise

    """
    ppr = gas_pseudored_pres(p, g)
    tpr = gas_pseudored_temp(t, g)
    eta1 = 0.0001 * (
        tpr * (28 + 48 * g - 5 * g**2) - 6.47 * g ** (-2) + 35 / g + 1.14 * g - 15.55
    )
    eta2diveta1 = (
        0.001
        * ppr
        * (
            (1057 - 8.08 * tpr) / ppr
            + (796 * np.power(ppr, 0.5) - 704) / (np.power(tpr - 1, 0.7) * (ppr + 1))
            - 3.24 * tpr
            - 38
        )
    )
    return eta2diveta1 * eta1


def oil_isothermal_density(rho, p):
    """Calculates the oil density for a given pressure at 15.6 degC

    B&W 1992 Equation 18

    Args:
        rho (array-like): The oil reference density (g/cc) at 15.6 degC
            can be compensated for disovled gases by running `oil_rho_sat` first.
        p (array-like): Pressure (MPa)

    Returns:
        (array-like): The oil density (g/cc) at pressure p
    """
    return (
        rho
        + (0.00277 * p - 1.71e-7 * np.power(p, 3)) * np.power(rho - 1.15, 2)
        + 3.49e-4 * p
    )


def oil_density(rho, p, t):
    """Calculates the oil density for given pressure and temperature

    For a live oil rho should be the output of `oil_rho_sat` to compensate
    for disolved gas.

    B&W 1992 Eq 19

    Args:
        rho (array-like): The oil density (g/cc)
        p (array-like): Pressure (MPa)
        t (array-like): Absolute temperature in degC

    Returns:
        (array-like): The oil density (g/cc) at pressure p and temperature ta
    """
    return oil_isothermal_density(rho, p) / (
        0.972 + 3.81e-4 * np.power(t + 17.78, 1.175)
    )


def oil_fvf(rho0, g, rs, t):
    """Calculate the oil formation volume factor b0

    Equation 23 from Batzle and Wang 1992
    All inputs must be broadcastable to an equivalent shape.

    Args:
        rho0 (array-like): The oil reference density (g/cc) at 15.6 degC
        g (array-like): The gas specific gravity
        rs (array-like): The Gas-to-Oil ratio Litre/litre
        t (array-like): In-situ temperature degC

    Returns:
        (array-like): The oil formation volume factor (FVF/b0)

    """
    # note: Dhannanjay 2006 shows 2.495 instead of 2.4 inside power bracket.
    return 0.972 + 0.00038 * np.power(2.4 * rs * np.sqrt(g / rho0) + t + 17.8, 1.175)


def oil_rg(oil, g, pres, temp, mode="rho"):
    """Calculate the Rg for oil with given paramters.

    Rg is the volume ratio of liberated gas to remaining oil at atmospheric
    pressure and 15.6degC. This covers equatsion 21a and 21b of B & W 1992.

    One input can be an array but if multiple inputs are arrays they must
    all have length n.

    Args:
        oil (array-like): The density of the oil at 15.6degC or the api of
            the oil if mode = 'api'
        g (array-like): gas specific gravity
        pres (array-like): reservoir pressure (MPa)
        temp (array-like): temperature (degC)
        mode (string): Default: 'rho', 'oil' argument mode:
            'rho' - oil input as density in g/cc
            'api' - oil input as API

    Returns:
        rg (array-like): (L/L)
    """
    if mode == "rho":
        a = 0.02123
        b = 4.072 / oil
    elif mode == "api":
        a = 2.03
        b = 0.02878 * oil
    else:
        raise KeyError("'mode must be either 'rho' or 'api'")
    return a * g * np.power(pres * np.exp(b - 0.00377 * temp), 1.205)


def oil_rho_sat(rho0, g, rg, b0):
    """Calculate the gas saturated oil density

    B&W Eq 24

    Args:
        rho0 (array-like): The oil reference density (g/cc) at 15.6 degC
        g (array-like): The gas specific gravity
        rg (array-like): The Gas-to-Oil ratio Litre/litre
        b0 (array-like): Oil formation volume factor FVF

    Returns:
        (array-like): The gas saturated oil density at 15.6 degC
    """
    return safe_divide((rho0 + 0.0012 * rg * g), b0)


def oil_rho_pseudo(rho0, rg, b0):
    """Calculates the oil pseudo density

    B&W 1992 Eq 22

    Args:
        rho0 (array-like): The oil reference density (g/cc) at 15.6 degC
        rg (array-like): Gas-to-Oil ratio Litre/litre
        b0 (array-like): Oil formation volume factor FVF

    Returns:
        (array-like): The pseudo-density of oil due to expansion from gas
    """
    return rho0 / (1 + 0.001 * rg) / b0


def oil_velocity(rho0, p, t, g, rg, b0=None):
    """Oil compressional-wave velocity

    B&W 1992 Eq 20a

    If b0 is None the B&W approximation will be used.
    The pseudo density for live oil is calculated intrinsically.

    Args:
        rho0 (array-like): The oil reference density (g/cc) at 15.6 degC
        p (array-like): In-situ pressure MPa
        t (array-like): In-situ temperature degC
        g (array-like): gas specific gravity
        rg (array-like): Gas-to-Oil ratio Litre/litre

    Returns:
        (array-like): The velocity of oil in m/s
    """
    if b0 is None:
        b0 = oil_fvf(rho0, g, rg, t)
    rhops = oil_rho_pseudo(rho0, rg, b0)
    return (
        2096 * np.sqrt(rhops / (2.6 - rhops))
        - 3.7 * t
        + 4.64 * p
        + 0.0115 * t * p * (4.12 * np.sqrt(1.08 / rhops - 1) - 1)
    )


def oil_bulkmod(rho, vp):
    """Oil bulk modulus GPa

    Args:
        rho (array-like): In-situ oil density
        vp (array-like): In-situ compressional velocity

    Returns:
        (array-like): Oil bulk modulus in GPa
    """
    return rho * np.power(vp, 2) * 1e-6


def _mwat_velocity_pure_sum(tl, pl, vl):
    """Pure Water Calc
    Input vectors should be type float64 to prevent overflow

    B&W Matrix Summation with water coefficients

    Args:
        tl (array-like): Temperature decC
        pl (array-like): Pressure MPa
        vl (array-like): Output velocity vector will be modified
    """
    for v, (tt, pp) in enumerate(zip(tl.ravel(), pl.ravel())):
        # using np.float64 may sacrifice some accuracy but was required for large powers
        t_ar = np.power(np.array([tt] * 5), np.arange(0, 5))
        p_ar = np.power(np.array([pp] * 4), np.arange(0, 4))
        p_ar = np.tile(p_ar, 5).reshape(5, 4)
        t_ar = t_ar.repeat(4).reshape(5, 4)
        vl[v] = np.sum(WATC * t_ar * p_ar)


@numba.njit
def _wat_velocity_pure_sum(tl, pl, vl):
    """Numba Pure Water Calc
    Input vectors should be type float64 to prevent overflow

    B&W Matrix Summation with water coefficients

    Args:
        tl (array-like): Temperature decC
        pl (array-like): Pressure MPa
        vl (array-like): Output velocity vector will be modified
    """
    for v, (tt, pp) in enumerate(zip(tl.ravel(), pl.ravel())):
        # using np.float64 may sacrifice some accuracy but was required for large powers
        for i in range(0, 5):
            for j in range(0, 4):
                vl[v] = vl[v] + WATC[i, j] * (tt**i) * (pp**j)


def wat_velocity_pure(t, p):
    """Returns the velocity of pure water at t and p
    t and p must have equal shape

    Args:
        t (array-like): temperature degC
        p (array-like): pressure MPa

    Returns:
        (array-like): compressional velocity of pure water m/s

    Notes:
        This approximation breaks down above pressures of 100MPa
    """
    # TODO: catch bad inputs - mismatched shapes
    tl = np.array(t)
    pl = np.array(p)
    if tl.shape == pl.shape:
        vl = np.zeros_like(tl)
    elif tl.shape == ():
        tl = np.full_like(pl, t)
        vl = np.zeros_like(pl)
    elif pl.shape == ():
        pl = np.full_like(tl, p)
        vl = np.zeros_like(tl)
    else:
        raise ValueError

    # vw = np.zeros_like(WATC)
    dims = vl.shape
    _wat_velocity_pure_sum(tl.astype(np.float64), pl.astype(np.float64), vl.ravel())
    vl.reshape(dims)
    try:
        return vl.item()
    except ValueError:
        return vl


def wat_velocity_brine(t, p, sal):
    """Velocity of a brine

    Args:
        t (array-like): temperature degC
        p (array-like): pressure MPa
        sal (array-like): weight fraction of salt
            note: Salinity (ppm) = fractional weight * 1E6

    Returns:
     (array-like): compressional velocity of brine m/s
    """
    return wat_velocity_pure(t, p) + (
        sal
        * (
            1170
            - 9.6 * t
            + 0.055 * np.power(t, 2)
            - 8.5e-5 * np.power(t, 3)
            + 2.6 * p
            + 0.0029 * t * p
            - 0.0476 * np.power(p, 2)
        )
        + np.power(sal, 1.5) * (780 - 10 * p + 0.16 * np.power(p, 2))
        - 1820 * np.power(sal, 2)
    )


def wat_density_pure(t, p):
    """Returns the density of pure water at t and p

    B&W 1992 Eq 27a

    Args:
        t (array-like): temperature degC
        p (array-like): pressure MPa

    Returns:
        (array-like): density of pure water g/cc
    """
    return 1 + 1e-6 * (
        -80 * t
        - 3.3 * np.power(t, 2)
        + 0.00175 * np.power(t, 3)
        + 489 * p
        - 2 * t * p
        + 0.016 * np.power(t, 2) * p
        - 1.3e-5 * np.power(t, 3) * p
        - 0.333 * np.power(p, 2)
        - 0.002 * t * np.power(p, 2)
    )


def wat_density_brine(t, p, sal):
    """Returns the density of brine at t and p

    B&W 1992 Eq 27b

    Args:
        t (array-like): temperature degC
        p (array-like): pressure MPa
        sal (array-like): weight fraction of salt
            note: Salinity (ppm) = fractional weight * 1E6

    Returns:
        (array-like): density of brine (g/cc)
    """
    return wat_density_pure(t, p) + sal * (
        0.668
        + 0.44 * sal
        + (1e-6)
        * (
            300 * p
            - 2400 * p * sal
            + t * (80 + 3 * t - 3300 * sal - 13 * p + 47 * p * sal)
        )
    )


def wat_salinity_brine(t: float, p: float, density: float) -> float:
    """Back out the salinity of a brine form it's density using root finding.

    Reverse B&W 1992 Eq 27b

    Args:
        t (float): temperature degC
        p (float): pressure MPa
        density (float): density of brine (g/cc)

    Returns:
        (float): weight fraction of salt; note: Salinity (ppm) = fractional weight * 1E6
    """

    def solve_density_for_salinity(sal):
        return density - wat_density_brine(t, p, sal)

    sol = root_scalar(
        solve_density_for_salinity, bracket=(0, 1), xtol=0.001, method="brentq"
    )
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Could not find solution for density to salinity")


def wat_bulkmod(rho: NDArrayOrFloat, vp: NDArrayOrFloat) -> NDArrayOrFloat:
    """Brine bulk modulus K

    Args:
        rho: water fluid density (g/cc)
        vp: water fluid compressional velocity (m/s)

    Returns:
        bulk modulus (GPa)
    """
    return rho * np.power(vp, 2) * 1e-6


def mixed_density(den, frac, *argv):
    """Returns the mixed fluid density based upon volume fractions

    Can take an arbitrary number of fluids and volume fractions.
    Volume fractions must sum to one for each sample.
    If the number of arguments is odd, the final frac is assumed to
        be the complement of all other fracs to sum to 1
    All array-like must be the same shape

    Args:
        den (array-like): first fluid density
        frac (array-like): [description]
        arg3 (array-like) : the next fluid density
        arg4 (array-like) : the next fluid fraction
        arg(3+2*n) (array-like) : the nth fluid density
        arg(3+2*n) (array-like) : the nth fluid fraction

    Returns:
        (array-like): mixed fluid density
    Notes:
    """
    # check arguments and find complement if necessary
    largv = len(argv)
    if largv % 2 == 1:
        fvol = np.array(frac)
        for i in range(1, largv, 2):
            fvol = fvol + np.array(argv[i])
        if np.any(fvol < 0.0) or np.any(fvol > 1.0):
            raise ValueError
        else:
            argv = argv + (1.0 - fvol,)

    # reset
    fvol = np.array(frac)
    rho = np.array(den) * np.array(frac)
    for i in range(0, largv, 2):
        rho = rho + np.array(argv[i]) * np.array(argv[i + 1])
        fvol = fvol + np.array(argv[i + 1])

    if np.allclose(fvol, 1):
        return rho
    else:
        return None  # change this to raise an appropriate error


def bulkmod(rho, vel):
    """Single phase fluid bulk modulus

    Args:
        rho (array-like): (g/cc)
        velp (array-like): material velocity (m/s)

    Returns:
        (array-like): material bulk modulus (GPa)
    """
    return rho * np.power(vel, 2) * 1e-6


def mixed_bulkmod(mod, frac, *argv):
    """Mixed fluid bulk modulus (Wood's Equation)


    Can take an arbitrary number of fluids and volume fractions.
    Volume fractions must sum to one for each sample.
    If the number of arguments is odd, the final frac is assumed to
        be the complement of all other fracs to sum to 1
    All array-like must be the same shape

    Args:
        mod (array-like): first fluid modulus
        frac (array-like): [description]
        arg3 (array-like) : the next fluid modulus
        arg4 (array-like) : the next fluid fraction
        arg(3+2*n) (array-like) : the nth fluid modulus
        arg(3+2*n) (array-like) : the nth fluid fraction

    Returns:
        (array-like): Wood's bulk modulus for a fluid mix
    """
    # check arguments and find complement if necessary
    largv = len(argv)
    if largv % 2 == 1:
        fvol = np.array(frac)
        for i in range(1, largv, 2):
            fvol = fvol + np.array(argv[i])
        if np.any(fvol < 0.0) or np.any(fvol > 1.0):
            raise ValueError
        else:
            argv = argv + (1.0 - fvol,)

    # reset
    fvol = np.array(frac)
    K = safe_divide(np.array(frac), np.array(mod))
    for i in range(0, largv, 2):
        K = K + np.array(argv[i + 1]) / np.array(argv[i])
        fvol = fvol + np.array(argv[i + 1])

    if np.allclose(fvol, 1):
        return safe_divide(1, K)
    else:
        return None  # change this to raise an appropriate error
