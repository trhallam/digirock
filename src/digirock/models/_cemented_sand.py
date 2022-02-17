"""Cemented Sand Model

Refs:
    Mavko et. al. (2009) The Rock Physics Handbook, Cambridge Press pp 255-257

"""

# pylint: disable=invalid-name

from numpy import power, pi
from ..elastic import poisson_ratio


def _alpha_scheme1(phi0, phi, ncontacts):
    """Scheme 1 of cemented sand model. Cement deposited at sand contacts.

    Defaults:
        phi0 = 0.36
        ncontacts = 9

    Args:
        phi0 (float): Sand only porosity
        phi (float): Rock porosity after cement added
        ncontacts (float): Number of contacts between grains.

    Returns:
        alpha (float): Ratio of cement radius to grain radius.
    """
    return 2 * power((phi0 - phi) / (3 * ncontacts * (1 - phi0)), 0.25)


def _alpha_scheme2(phi0, phi):
    """Scheme 2 of cemented sand model. Cement deposited evenly around grains.

    Defaults:
        phi0 = 0.36

    Args:
        phi0 (float): Sand only porosity
        phi (float): Rock porosity after cement added

    Returns:
        alpha (float): Ratio of cement radius to grain radius.
    """
    return power(2 * (phi0 - phi) / (3 * (1 - phi0)), 0.5)


def _cemented_sand_alpha(phi0, phi, ncontacts=None, alpha="scheme1"):
    """Alpha calculation for keyword."""
    if alpha == "scheme1" and ncontacts is not None:
        return _alpha_scheme1(phi0, phi, ncontacts)
    if alpha == "scheme2":
        return _alpha_scheme2(phi0, phi)
    if isinstance(alpha, float):
        return alpha

    raise ValueError("Unknown alpha or missing ncontacts for scheme1")


def dryframe_cemented_sand(
    k_sand, mu_sand, k_cem, mu_cem, phi0, phi, ncontacts, alpha="scheme1"
):
    """Dryframe moduli for Cemented-Sand model.

    Assumes sand grains are in contact and cement fills the porespace between grains.

    Scheme 1: Cement preferentially fills spaces around contacts increasing shear strength.
    Scheme 2: Cement covers grains evenly but maintains contacts between sand grains.
    Alpha as float: Specify a custom value for alpha.

    Args:
        k_sand (array-like): Bulk modulus of sand grains.
        mu_sand (array-like): Shear modulus of sand grains.
        k_cem (array-like): Bulk modulus of cement.
        mu_cem (array-like): Shear modulus of cement.
        phi0 (array-like): Material porosity without any cement. For perfect spherical grains
            phi0 = 0.36
        phi (array-like): Material porosity after substituting in cememnt to porespace.
        ncontacts (float): Number of contacts between grains.
        alpha (str/float, Optional): The cement fill scheme to use. Defaults to 'scheme1.
            One of ['sheme1', 'scheme2', float]

    Returns:
        (array-like, array-like): Bulk and shear modului of cemented-sand.
    """
    pois_sand = poisson_ratio(k_sand, mu_sand)
    pois_cem = poisson_ratio(k_cem, mu_cem)
    alpha = _cemented_sand_alpha(phi0, phi, ncontacts, alpha)

    lam_tau = mu_cem / (pi * mu_sand)
    lam_n = (2 * mu_cem * (1 - pois_sand) * (1 - pois_cem)) / (
        pi * mu_sand * (1 - 2 * pois_cem)
    )
    pois_sand2 = power(pois_sand, 2)
    c_tau = (
        10e-4
        * (9.654 * pois_sand2 + 4.945 * pois_sand + 3.1)
        * power(lam_tau, 0.01867 * pois_sand2 + 0.4011 * pois_sand - 1.8186)
    )
    b_tau = (0.0573 * pois_sand2 + 0.0937 * pois_sand + 0.202) * power(
        lam_tau, 0.0274 * pois_sand2 + 0.0529 * pois_sand - 0.8765
    )
    a_tau = (
        -0.01
        * (2.26 * pois_sand2 + 2.07 * pois_sand + 2.3)
        * power(lam_tau, 0.079 * pois_sand2 + 0.1754 * pois_sand - 1.342)
    )
    s_tau = a_tau * power(alpha, 2) + b_tau * alpha + c_tau
    c_n = 0.00024649 * power(lam_n, -1.9864)
    b_n = 0.20405 * power(lam_n, -0.89008)
    a_n = -0.024153 * power(lam_n, -1.3646)
    s_n = a_n * power(alpha, 2) + b_n * alpha + c_n

    keff = (ncontacts * k_cem * s_n / 6) * (1 - phi0)
    return (keff, 3 * keff / 5 + (3 * ncontacts * mu_cem * s_tau / 20.0) * (1 - phi0))
