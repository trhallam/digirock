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

import numpy as np

# pylint: disable=unused-import
from ..utils import safe_divide


def _vr_bound(m, f, bound):
    """Common function for voight_upper and reuss_lower

    Args:
        m (array-like): Modulus
        f (array-like): Volume fraction
        bound (str): Bound, one of ['voigt', 'reuss']
    """
    m = np.array(m).squeeze()
    f = np.array(f).squeeze()
    if m.shape[-1] == f.shape[-1]:
        if np.allclose(np.sum(f, axis=-1), 1.0):
            if bound == "voigt":
                return np.sum(np.multiply(m, f), axis=-1)
            elif bound == "reuss":
                return 1.0 / np.sum(np.divide(f, m), axis=-1)
        else:
            raise ValueError(
                f"Input f must sum to 1.0 current sum is {np.sum(f, axis=-1)}"
            )
    else:
        raise ValueError("Inputs m and f must have equal length")


def voigt_upper_bound(m, f):
    """Calculate the Voigt upper bound for material moduli

    Args:
        m (array_like): the material moduli
        f (array_like): the material volume fractions where sum(f, axis=-1) = 1

    Returns:
        (): the Voigt upper bound modulus
    """
    return _vr_bound(m, f, "voigt")


def reuss_lower_bound(m, f):
    """Calculate the Reuss lower bound for material moduli

    Args:
        m: the material moduli
        f: the material volume fractions where sum(f) = 1

    Returns:
        (): the Reuss lower vound modulus
    """
    return _vr_bound(m, f, "reuss")


def vrh_avg(mclay, mnonclay, vclay):
    """Calculates the Voigt-Reuss-Hill mix for a simple other/shale mix.

    Args:
        mclay (array-like): clay component moduli
        mnonclay (array-like): non-clay component moduli
        vclay (array-like): fraction of material that is clay where 0 < fclay < 1

    Returns:
        (array-like): Voigt-Reuss-Hill average modulus
    """
    fclay = vclay
    fnonclay = 1 - vclay
    return 0.5 * (
        voigt_upper_bound([mnonclay, mclay], np.transpose([fnonclay, fclay]))
        + reuss_lower_bound([mnonclay, mclay], np.transpose([fnonclay, fclay]))
    )


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
