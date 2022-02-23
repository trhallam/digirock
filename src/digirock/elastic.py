"""Core functions for elastic property modelling.
"""
from typing import Tuple
import numpy as np

from .utils.types import NDArrayOrFloat
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
def acoustic_moduli(
    velp: NDArrayOrFloat, vels: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
    """Calculate the bulk and shear modulus for a material from acoustic properties

    Whether from dynamic velocity measurements or from wireline log data, we can relate the bulk
    modulus ($\kappa$) of a rock to it's acoustic properties. If the rock is saturated then the modulus will
    be $\kappa_{sat}$, if the rock is measured while dry then the modulus will approximate the porous rock
    framework $\kappa_{dry}$.

    $$
    \kappa = \\rho_b \\left( v_p^2 - \\frac{4v_s^2}{3} \\right)
    $$

    $$
    \mu = \\rho_b v_s^2
    $$

    Args:
        velp: Compressional velocity (m/s)
        vels: Shear velocity (m/s)
        rhob: Bulk density (g/cc)

    Returns:
        acoustic bulk modulus GPa, acoustic shear modulus GPa

    References:
        [3] Smith et al. 2003
    """
    return (
        (rhob * (np.power(velp, 2) - (4 / 3) * np.power(vels, 2))) * 1e-6,
        rhob * np.power(vels, 2) * 1e-6,
    )


@broadcastable("k", "mu", "rhob")
def acoustic_vel(
    k: NDArrayOrFloat, mu: NDArrayOrFloat, rhob: NDArrayOrFloat
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
    """Calculate the compressional $v_p$ and shear $v_s$ velocity from material bulk $\kappa$ and shear $\mu$ moduli and density $\\rho_b$.

    $$
    v_p = \\sqrt{\\frac{\kappa + \\dfrac{4}{3}\mu}{\\rho_b}}
    $$

    $$
    v_s = \\sqrt{\\frac{\mu}{\\rho_b}}
    $$

    Args:
        k: bulk modulus GPa
        mu: shear modulus GPa
        rhob: bulk density g/cc

    Returns:
        compressional velocity (m/s), shear velocity (m/s)
    """
    return 1000 * np.sqrt((k + (4 / 3) * mu) / rhob), 1000 * np.sqrt(mu / rhob)
