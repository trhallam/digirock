"""Hashin-Shtrikman-Walpole Models for moduli mixing.

Refs:
    Mavko et. al. (2009) The Rock Physics Handbook, Cambridge Press
"""

# pylint: disable=invalid-name

from numpy import r_, full_like, asarray, where, zeros
from numpy import sum as npsum

from ..utils import safe_divide


def hs_kbounds(k1, k2, mu1, f1):
    """Calculate the Hashin-Shtrikman bulk moduli bounds for an isotropic elastic mixture

    Upper and lower bounds are computed by interchanging which constituent material is
    subscripted 1 and which is subscripted 2. In general
        Upper bound -> stiffest material is subscripted 1
        Lower bound -> stiffest material is subscripted 1

    Args:
        k1: constituent one bulk modulus
        k2: constituent two bulk modulus
        mu1: constituent one shear modulus
        f1: material one volume fraction 0 <= f1 <= 1

    Returns:
        (): HS bulk modulus bounds

    """
    return k1 + (1 - f1) / (1 / (k2 - k1) + f1 / (k1 + 4 * mu1 / 3))


def hs_mubounds(k1, mu1, mu2, f1):
    """Calculate the Hashin-Shtrikman shear moduli bounds for an isotropic elastic mixture

    Upper and lower bounds are computed by interchanging which constituent material is
    subscripted 1 and which is subscripted 2. In general
        Upper bound -> stiffest material is subscripted 1
        Lower bound -> stiffest material is subscripted 1

    Args:
        k1: constituent one bulk modulus
        mu1: constituent one shear modulus
        mu2: constituent two shear modulus
        f1: material one volume fraction 0 <= f1 <= 1

    Return:
        (): HS shear modulus bounds (positive)

    """
    return mu1 + (1 - f1) / (
        (1 / (mu2 - mu1)) + 2 * f1 * (k1 + 2 * mu1) / (5 * mu1 * (k1 + 4 * mu1 / 3))
    )


def _hsw_medium_avg(z, zscale, *args):
    z = asarray(z)
    nargs = len(args)
    dims = None
    size = None
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (float, int)):
            args[i] = r_[arg]
        elif dims is None:
            dims = arg.shape
            size = arg.size
        elif arg.shape != dims:
            raise ValueError("inputs must have matching dimensions")

    bavg = zeros((nargs // 2, size))
    for i, (comp, frac) in enumerate(zip(args[::2], args[1::2])):
        bavg[i, :] = safe_divide(frac.ravel(), (comp.ravel() + zscale * z.ravel()))
    return 1 / npsum(bavg, axis=0).reshape(dims) - zscale * z


def _hsw_lambda(z, *args):
    """Hashin-Shtrikman Lamba component for bulk moduli.

    Works for an arbitrary number of minerals/fluids.

    Args:
        z (float): argument of Lambda (minimum or maximum component shear modulus)
        args (array-like): pairs of mineral moduli arrays and volume fraction arrays

    Refs:
        Rock Physics Handbook p171
    """
    return _hsw_medium_avg(z, 4.0 / 3.0, *args)


def _hsw_zeta(k, mu):
    """Scalar for _hsw_gamma

    k and mu are usally either the mixing component min or max values for bulk and shear modulus.

    Args:
        k (array-like): Bulk modulus
        mu (array-like): Shear modulus

    Returns:
        array-like: Scalar for z input to _hsw_gamma

    Refs:
        Rock Physics Handbook p171
    """
    return mu / 6 * ((9 * k + 8 * mu) / (k + 2 * mu))


def _hsw_gamma(z, *args):
    """Shear modulus bounds for Hashin-Shtikman-Walpole method.

    Args:
        z (array-like): Output of _hs_gamma
        args (array-like): pairs of mineral moduli arrays and volume fraction arrays

    Returns:
       array-like: The upper or lower shear modulus bound depending upon z.

    Refs:
        Rock Physics Handbook p171
    """
    gamma = _hsw_medium_avg(z, 1.0, *args)
    if asarray(z).size == 1:
        z = full_like(gamma, z)
    return where(z == 0.0, 0.0, gamma)


def hsw_avg_bulk_modulus(mumin, mumax, *args):
    """Hashin-Shrikman-Walpole average bulk modulus for N materials.

    Args:
        mumin (array-like): Minimum material moduli.
        mumax (array-like): Maximum material moduli.
        args (array-like): pairs of mineral moduli arrays and volume fraction arrays
    """
    return 0.5 * (_hsw_lambda(mumax, *args) + _hsw_lambda(mumin, *args))


def hsw_avg_shear_modulus(kmax, mumax, kmin, mumin, *args):
    """Hashin-Shrikman-Walpole average shear modulus for N materials.

    Args:
        kmax (array-like): Maximum material bulk moduli
        mumax (array-like): Minimum material shear moduli
        kmin (array-like): Maximum material bulk moduli
        mumin (array-like): Minimum materal shear moduli.
        args (array-like): pairs of mineral moduli arrays and volume fraction arrays

    Returns:
        array-like: The average of the upper and lower bounds for the input args.
    """
    zmin = _hsw_zeta(kmin, mumin)
    zmax = _hsw_zeta(kmax, mumax)
    return 0.5 * (_hsw_gamma(zmin, *args) + _hsw_gamma(zmax, *args))
