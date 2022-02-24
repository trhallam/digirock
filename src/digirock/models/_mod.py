# -*- coding: utf8 -*-
# pylint: disable=invalid-name

"""Functions related to the moduli and mixed moduli of minerals

This module contains functions related to the calculation of

 1. modulus bounds
 2. moduli for mixed mineral materials
 3. Gassmann fluid substitution

Refs:
    MacBeth, 2004, A classification for the pressure-sensitivity properties of a sandstone rock frame\n
    Smith, Sondergeld and Rai, 2003, Gassmann fluid substitutions: A tutorial, Geophysics, 68, pp430-440
"""
from more_itertools import chunked
import numpy as np

# pylint: disable=unused-import
from ..utils._utils import safe_divide, _process_vfrac
from ..typing import NDArrayOrFloat


def voigt_upper_bound(
    mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculate the Voigt upper bound for material moduli

    Args:
        mod: first fluid modulus
        frac: first fluid volume fraction
        argv: additional fluid and volume fraction pairs.

    Returns:
        Voigt upper bound for modulus mix
    """
    args = _process_vfrac(*((mod, frac) + argv))
    mod_sum = 0
    for modi, vfraci in chunked(args, 2):
        mod_sum = mod_sum + np.array(vfraci) * np.array(modi)
    return mod_sum


def reuss_lower_bound(
    mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculate the Reuss lower bound for material moduli

    Args:
        mod: first fluid modulus
        frac: first fluid volume fraction
        argv: additional fluid and volume fraction pairs.

    Returns:
        Reuss upper bound for modulus mix
    """
    args = _process_vfrac(*((mod, frac) + argv))
    mod_sum = 0
    for modi, vfraci in chunked(args, 2):
        mod_sum = mod_sum + safe_divide(np.array(vfraci), np.array(modi))
    return safe_divide(1, mod_sum)


def vrh_avg(
    mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculates the Voigt-Reuss-Hill average for a multi-modulus mix.

    Args:
        mod: first modulus
        frac: first volume fraction
        argv: additional modulus and volume fraction pairs.

    Returns:
        Voigt-Reuss-Hill average modulus
    """
    vub = voigt_upper_bound(mod, frac, *argv)
    rlb = reuss_lower_bound(mod, frac, *argv)
    return 0.5 * (vub + rlb)


def mixed_density(
    den: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Mixed density $\\rho_M$ based upon volume fractions

    $$
    \\rho_M = \sum_{i=1}^N \\phi_i \\rho_i
    $$

    Can take an arbitrary number of denisties $\\rho_i$ and volume fractions $\\phi_i$.
    Volume fractions must sum to one. If the number of arguments is odd, the final
    volume fraction is assumed to be the ones complement of all other fractions.

    Inputs must be [broadcastable](https://numpy.org/doc/stable/user/basics.broadcasting.html).

    `argv` as the form `(component_1, vfrac_1, component_2, vfrac2, ...)` or to use the complement for the final
    component `(component_1, vfrac_1, component_2)`

    Args:
        den: first fluid density
        frac: first fluid volume fraction
        argv: additional fluid and volume fraction pairs.

    Returns:
        mixed density for combined material
    """
    args = _process_vfrac(*((den, frac) + argv))
    den_sum = 0
    for deni, vfraci in chunked(args, 2):
        den_sum = den_sum + np.array(vfraci) * np.array(deni)
    return den_sum


def dryframe_delta_pres(erp_init, erp, mod_vrh, mod_e, mod_p, phi, phic):
    """Calculates the dry rock frame for a given stress regieme and depth by factoring the
    difference between two pressure regiemes.

    Args:
        erp_init (array_like): Effective Initial Reservoir Pressure (MPa)
                                = Overburden Pressure - Initial Reservoir Pressure
        erp (array_like): Effective Current Reservoir Pressure (MPa)
                                = Overburden Pressure - Current Reservoir Pressure
        mod_vrh: Voigt-Reuss-Hill average modulus (check this).
        mod_e: modulus stress sensitivity metric *2
        mod_p: modulus characteristic pressure constant *2
        phi: rock porosity
        c (list: Default:None): If 'c=None' constant porosity is used else c is a critical
            porosity (phic) list of length 5 e.g. [c1, c2, c3, c4, c5] where when phi < c3;
            phic = c1 + c2*phi and when phi >= c3 phic = c3 + c4*phi

    Returns:
        moddry (array_like): the dry-frame modulus for inputs

    References:
        [1] Amini and Alvarez (2014)
        [2] MacBeth (2004)
    """
    # Calcuate Bulk Modulus for Dry Frame
    dry1 = np.where(phi >= phic, 0.0, mod_vrh * (1 - phi / phic))
    moddry = (
        dry1
        * (1 + (mod_e * np.exp(-erp_init / mod_p)))
        / (1 + (mod_e * np.exp(-erp / mod_p)))
    )
    return moddry


def dryframe_dpres(dry_mod, pres1, pres2, sse, ssp):
    """Calculates the dry rock frame for a given stress regieme and depth by factoring the
    difference between two pressure regiemes.

    Args:
        dry_mod (array-like): Dryframe modulus (MPa)
        pres1 (array-like): Pressure calibrated to dryframe modulus. (MPa)
        pres2 (array-like): Pressure of output (MPa)
        sse (float): Stress-sensitivity parameter E
        ssp (float): Stress-sensitivity parameter P

    Returns:
        array_like: the dry-frame modulus adjusted to pressure (pres2)

    References:
        [1] Amini and Alvarez (2014)
        [2] MacBeth (2004)
    """
    return (
        dry_mod
        * (1 + (sse * np.exp(-pres1 / ssp)))
        / (1 + (sse * np.exp(-pres2 / ssp)))
    )


def dryframe_stress(mod_e, mod_p, inf, p):
    """Dry frame bulk modulus from stress sensitivity coefficients.

    Based on MacBeth 2004 for sandstone rock frame.

    Args:
        mod_e (array-like): modulus stress sensitivity metric *2 MPa
        mod_p (array-like): modulus characteristic pressure constant *2 MPa
        inf (array-like): modulus background high pressure asymptote MPa
        p (array-like): the effective pressure MPa
    """
    return inf / (1 + (mod_e * np.exp(-p / mod_p)))


def dryframe_acoustic(ksat, kfl, k0, phi):
    """Dry frame bulk modulus from material saturated modulus

    Args:
        ksat (array_like): saturated bulk modulus
        kfl (array_like): fluid bulk modulus
        k0 (array_like): matrix bulk modulus
        phi (array_like): porosity (frac)

    Returns:
        kdry (array_like): backed-out dry fame bulk modulus (inverse Gassman_fluidsub)
    """
    return (ksat * (phi * k0 / kfl + 1 - phi) - k0) / (
        phi * k0 / kfl + ksat / k0 - 1 - phi
    )


def dryframe_density(rhob, rhofl, phi):
    """Dry frame grain density

    Primarily used for fluid substitution.

    Args:
        rhob (array-like): saturated rock bulk modulus
        rhofl (array-like): fluid density
        phi (array-like): porosity (frac)

    Returns:
        (array-like): dry frame grain density
    """
    return (rhob - rhofl * phi) / (1 - phi)


def saturated_density(rhog, rhofl, phi):
    """Saturated bulk density

    Args:
        rhog (array-like): dry frame grain density
        rhofl (array-like): fluid density
        phi (array-like): porosity (frac)

    Returns:
        (array-like): saturated rock bulk density (rhob)
    """
    return rhog * (1 - phi) + rhofl * phi


def gassmann_fluidsub(kdry, kfl, k0, phi):
    """Gassmann fluid substitution for saturated rock bulk modulus

    Gassmann fluid substitution assumes:
        The rock is homogenous and isotropic, rocks with diverse mineral acoustic properties will
            violate this assumption.
        All the porespace is connected, this is propably not true for low porosity rocks.
        The frequencies being used to measure the rock properties fall within the seismic
            bandwidth. Logging frequencies are usually ok for high-porosity clean sands.
        Homogeonous mixing of the fluid. This is reasonable over geologic time but may be
            inappropriate for production timescales.

    Args:
        kdry (array_like): dry frame rock bulk modulus
        kfl (array_like): fluid bulk modulus
        k0 (array_like): matrix bulk modulus
        phi (array_like): porosity (frac)

    Returns:
        ksat (array_like): saturated rock bulk modulus
    """
    return kdry + np.power(1 - kdry / k0, 2) / (
        safe_divide(phi, kfl)
        + safe_divide((1 - phi), k0)
        - safe_divide(kdry, np.power(k0, 2))
    )


def patchy_fluidsub(kdry, mu, kfls, vfrac):
    """Patch fluid substitution for non-homogenous fluid mixing

    Args:
        kdry (array_like): dry frame rock bulk modulus
        mu (array_like): rock shear modulus
        kfls (array_like): list or array of length M fluid moduli
        vfrac (array_like): list or array of length M or M-1 fluid volume fractions
                                for length M sum(vfrac) == 1
                                for length M-1  1 - sum(vfrac) = complement for kfls[-1]

    Returns:
        ksat (array_like): the saturated rock bulk modulus
    """
    # TODO make this more matrix oriented
    nfl = len(kfls)
    if len(vfrac) == nfl - 1:
        pass
    for fl in kfls:
        pass
