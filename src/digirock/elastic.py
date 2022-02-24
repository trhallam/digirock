"""Core functions for elastic property modelling.
"""
from typing import Tuple
import numpy as np

from .typing import NDArrayOrFloat
from .utils._decorators import broadcastable


@broadcastable("k", "mu")
def poisson_ratio(k: NDArrayOrFloat, mu: NDArrayOrFloat) -> NDArrayOrFloat:
    """Poisson's Ratio $v$ from bulk $\kappa$ and shear $\mu$ moduli.

    $$
    v = \\frac{3\kappa - 2\mu}{2(3\kappa + \mu)}
    $$

    Args:
        k: Bulk modulus
        mu: Shear modulus

    Returns:
        Poisson's ratio
    """
    return (3 * k - 2 * mu) / (2 * (3 * k + mu))


@broadcastable("velp", "vels", "rhob")
def acoustic_bulk_moduli(
    velp: NDArrayOrFloat, vels: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculate the bulk modulus for a material from acoustic properties

    Whether from dynamic velocity measurements or from wireline log data, we can relate the bulk
    modulus ($\kappa$) of a rock to it's acoustic properties. If the rock is saturated then the modulus will
    be $\kappa_{sat}$, if the rock is measured while dry then the modulus will approximate the porous rock
    framework $\kappa_{dry}$.

    $$
    \kappa = \\rho_b \\left( v_p^2 - \\frac{4v_s^2}{3} \\right)
    $$

    Args:
        velp: Compressional velocity (m/s)
        vels: Shear velocity (m/s)
        rhob: Bulk density (g/cc)

    Returns:
        acoustic bulk modulus GPa

    References:
        [3] Smith et al. 2003
    """
    return rhob * (np.power(velp, 2) - (4 / 3) * np.power(vels, 2)) * 1e-6


@broadcastable("vels", "rhob")
def acoustic_shear_moduli(vels: NDArrayOrFloat, rhob: NDArrayOrFloat) -> NDArrayOrFloat:
    """Calculate the shear modulus for a material from acoustic properties

    $$
    \mu = \\rho_b v_s^2
    $$

    Args:
        vels: Shear velocity (m/s)
        rhob: Bulk density (g/cc)

    Returns:
        acoustic shear modulus GPa

    References:
        [3] Smith et al. 2003
    """
    return rhob * np.power(vels, 2) * 1e-6


@broadcastable("velp", "vels", "rhob")
def acoustic_moduli(
    velp: NDArrayOrFloat, vels: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
    """Shortcut for getting both acoustic moduli

    Args:
        velp: Compressional velocity (m/s)
        vels: Shear velocity (m/s)
        rhob: Bulk density (g/cc)

    Returns:
        acoustic bulk modulus GPa, acoustic shear modulus GPa

    References:
        [3] Smith et al. 2003
    """
    return acoustic_bulk_moduli(velp, vels, rhob), acoustic_shear_moduli(vels, rhob)


@broadcastable("k", "mu", "rhob")
def acoustic_velp(
    k: NDArrayOrFloat, mu: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculate the compressional $v_p$ from material bulk $\kappa$ and shear $\mu$ moduli and density $\\rho_b$.

    $$
    v_p = \\sqrt{\\frac{\kappa + \\dfrac{4}{3}\mu}{\\rho_b}}
    $$

    Args:
        k: bulk modulus GPa
        mu: shear modulus GPa
        rhob: bulk density g/cc

    Returns:
        compressional velocity (m/s), shear velocity (m/s)
    """
    return 1000 * np.sqrt((k + (4 / 3) * mu) / rhob)


@broadcastable("k", "mu", "rhob")
def acoustic_vels(mu: NDArrayOrFloat, rhob: NDArrayOrFloat) -> NDArrayOrFloat:
    """Calculate the shear $v_s$ velocity from material shear $\mu$ moduli and density $\\rho_b$.

    $$
    v_s = \\sqrt{\\frac{\mu}{\\rho_b}}
    $$

    Args:
        mu: shear modulus GPa
        rhob: bulk density g/cc

    Returns:
        shear velocity (m/s)
    """
    return 1000 * np.sqrt(mu / rhob)


@broadcastable("k", "mu", "rhob")
def acoustic_vel(
    k: NDArrayOrFloat, mu: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
    """Shortcut for getting both velocities

    Args:
        k: bulk modulus GPa
        mu: shear modulus GPa
        rhob: bulk density g/cc

    Returns:
        compressional velocity (m/s), shear velocity (m/s)
    """
    return acoustic_velp(k, mu, rhob), acoustic_vels(mu, rhob)
