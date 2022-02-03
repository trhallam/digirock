"""Core functions for PEM modelling.
"""

# pylint: disable=invalid-name

import numpy as np


def poisson_moduli(k, mu):
    """Poisson's Ratio from material moduli.

    Args:
        k (array-like): Bulk modulus
        mu (array-like): Shear modulus

    Returns:
        (array-like): poisson's ratio
    """
    return (3 * k - 2 * mu) / (2 * (3 * k + mu))


def acoustic_moduli(velp, vels, rhob):
    """Calculate the bulk and shear modulus for a material from acoustic properties

    Whether from dynamic velocity measurements or from wireline log data, we can relate the bulk
    modulus of a rock to it's acoustic properties. If the rock is saturated then the modulus will
    be Ksat, if the rock is measured while dry then the modulus will approximate the porous rock
    framework Kdry.

    Args:
        velp (array_like): Compressional velocity (m/s)
        vels (array_like): Shear velocity (m/s)
        rhob (array_like): Bulk density (g/cc)

    Returns:
        (array_like): acoustic bulk modulus GPa
        (array_like): acoustic shear modulus GPa

    References:
        [3] Smith et al. 2003
    """
    return (
        (rhob * (np.power(velp, 2) - (4 / 3) * np.power(vels, 2))) * 1e-6,
        rhob * np.power(vels, 2) * 1e-6,
    )


def acoustic_vel(k, mu, rhob):
    """Calculate the compressional and shear velocity from material bulk moduli and density

    Args:
        k (array_like): bulk modulus GPa
        mu (aray_like): shear modulus GPa
        rhob (array_like): bulk density g/cc

    Returns:
        velp (array_like): compressional velocity (m/s)
        vels (array_like): shear velocity (m/s)
    """
    return 1000 * np.sqrt((k + (4 / 3) * mu) / rhob), 1000 * np.sqrt(mu / rhob)
