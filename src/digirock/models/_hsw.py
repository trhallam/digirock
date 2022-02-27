"""Hashin-Shtrikman-Walpole Models for moduli mixing.

Refs:
    Mavko et. al. (2009) The Rock Physics Handbook, Cambridge Press
"""

# pylint: disable=invalid-name
from typing import Tuple
from more_itertools import chunked
import numpy as np

from ..typing import NDArrayOrFloat
from ..utils._utils import _process_vfrac, nan_divide, check_broadcastable


def hs_kbounds2(
    k1: NDArrayOrFloat, k2: NDArrayOrFloat, mu1: NDArrayOrFloat, f1
) -> NDArrayOrFloat:
    """Calculate the Hashin-Shtrikman bulk moduli bounds for an isotropic elastic mixture

    Upper and lower bounds are computed by interchanging which constituent material is
    subscripted 1 and which is subscripted 2. In general:

     - Upper bound -> stiffest material is subscripted 1
     - Lower bound -> stiffest material is subscripted 1

    Args:
        k1: constituent one bulk modulus
        k2: constituent two bulk modulus
        mu1: constituent one shear modulus
        f1: material one volume fraction 0 <= f1 <= 1

    Returns:
        HS bulk modulus bounds

    Refs:
        Rock Physics Handbook p170
    """
    return k1 + (1 - f1) / (1 / (k2 - k1) + f1 / (k1 + 4 * mu1 / 3))


def hs_mubounds2(
    k1: NDArrayOrFloat, mu1: NDArrayOrFloat, mu2: NDArrayOrFloat, f1: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Calculate the Hashin-Shtrikman shear moduli bounds for an isotropic elastic mixture

    Upper and lower bounds are computed by interchanging which constituent material is
    subscripted 1 and which is subscripted 2. In general:

     - Upper bound -> stiffest material is subscripted 1
     - Lower bound -> stiffest material is subscripted 1

    Args:
        k1: constituent one bulk modulus
        mu1: constituent one shear modulus
        mu2: constituent two shear modulus
        f1: material one volume fraction 0 <= f1 <= 1

    Return:
        HS shear modulus bounds (positive)

    Refs:
        Rock Physics Handbook p170
    """
    return mu1 + (1 - f1) / (
        (1 / (mu2 - mu1)) + 2 * f1 * (k1 + 2 * mu1) / (5 * mu1 * (k1 + 4 * mu1 / 3))
    )


def _hsw_get_minmax(
    mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat, minmax="min"
) -> NDArrayOrFloat:
    """Get the minimum or maximum modulus but not when relevant frac is zero."""
    args = _process_vfrac(*(mod, frac) + argv)
    to_shp = check_broadcastable(**{f"arg{i}": arg for i, arg in enumerate(args)})
    arg_list = list()
    for modi, vfraci in chunked(args, 2):
        modi = np.where(vfraci == 0, np.nan, modi)
        arg_list.append(modi)
    out = np.stack(tuple(np.broadcast_to(modi, to_shp) for modi in arg_list))
    if minmax == "min":
        return np.nanmin(out, axis=0)
    elif minmax == "max":
        return np.nanmax(out, axis=0)
    else:
        raise ValueError("Unknown value for minmax")


def _hsw_bulk_modulus_avg(
    z: NDArrayOrFloat, mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Averaging part HSW bulk modulus medium $\\Lambda$ equation.

    Args:
        z: minimum modulus
        mod: first modulus
        frac: first volume fraction
        argv: additional modulus and volume fraction pairs.

    Returns:
        Hashin-Stricktman
        frac
        argv

    Refs:
        Rock Physics Handbook p171
    """
    args = _process_vfrac(*(mod, frac) + argv)
    mod_sum = 0
    for modi, vfraci in chunked(args, 2):
        mod_sum = mod_sum + nan_divide(vfraci, modi + 4 * z / 3)
    out = nan_divide(1, mod_sum) - 4 * z / 3
    out[np.isnan(out)] = 0.0
    return out


def _hsw_shear_modulus_avg(
    z: NDArrayOrFloat, mod: NDArrayOrFloat, frac: NDArrayOrFloat, *argv: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Averaging part HSW shear modulus medium $\\Gamma$ equation.

    Args:
        z: minimum modulus
        mod: first modulus
        frac: first volume fraction
        argv: additional modulus and volume fraction pairs.

    Returns:
        Hashin-Shtrikman

    Refs:
        Rock Physics Handbook p171
    """
    args = _process_vfrac(*(mod, frac) + argv)
    mod_sum = 0
    for modi, vfraci in chunked(args, 2):
        mod_sum = mod_sum + nan_divide(vfraci, modi + z)
    out = nan_divide(1, mod_sum) - z
    out[np.isnan(out)] = 0.0
    return out


def _hsw_zeta(k: NDArrayOrFloat, mu: NDArrayOrFloat) -> NDArrayOrFloat:
    """Scalar for _hsw_shear_modulus_avg

    k and mu are usually either the mixing component min or max values for bulk and shear modulus.

    Args:
        k: Bulk modulus
        mu: Shear modulus

    Returns:
        array-like: Scalar for z input to _hsw_gamma

    Refs:
        Rock Physics Handbook p171
    """
    out = mu * nan_divide(9 * k + 8 * mu, k + 2 * mu) / 6
    out[np.isnan(out)] = 0.0
    return out


def hsw_bounds(
    bulk_mod: NDArrayOrFloat,
    shear_mod: NDArrayOrFloat,
    frac: NDArrayOrFloat,
    *argv: NDArrayOrFloat,
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat, NDArrayOrFloat, NDArrayOrFloat]:
    """Hashin-Shtrikman-Walpole bounds

    $$
    K^{HS^+} = \\Lambda(\\mu_{max})
    $$

    Args:
        bulk_mod: first component bulk modulus
        shear_mod: first component shear modulus
        frac: first component volume fraction
        *argv: additional triplets of bulk_mod, shear_mod and frac for additional components

    Returns:
        K_hsp, K_hsm, Mu_hsp, Mu_hsm


    Refs:
        Rock Physics Handbook p171-172

    This example comes form the RPH p172

    Examples:
    ```
    hsw_bounds(35, 45, (1-0.27)*0.8, 75, 31, (1-0.27)*0.2, 2.2, 0) # porosity is 0.27
    >>> (array([26.43276985]), array([7.07415429]), array([24.61588052]), array([0.]))
    ```
    """
    args = _process_vfrac(*(bulk_mod, shear_mod, frac) + argv, i=2)
    shear_min = _hsw_get_minmax(
        *(v for i, v in enumerate(args) if i % 3 != 0), minmax="min"
    )
    shear_max = _hsw_get_minmax(
        *(v for i, v in enumerate(args) if i % 3 != 0), minmax="max"
    )
    bulk_min = _hsw_get_minmax(
        *(v for i, v in enumerate(args) if i % 3 != 1), minmax="min"
    )
    bulk_max = _hsw_get_minmax(
        *(v for i, v in enumerate(args) if i % 3 != 1), minmax="max"
    )
    k_hsp = _hsw_bulk_modulus_avg(
        shear_max, *(v for i, v in enumerate(args) if i % 3 != 1)
    )
    k_hsm = _hsw_bulk_modulus_avg(
        shear_min, *(v for i, v in enumerate(args) if i % 3 != 1)
    )
    zeta_hsp = _hsw_zeta(bulk_max, shear_max)
    zeta_hsm = _hsw_zeta(bulk_min, shear_min)
    mu_hsp = _hsw_shear_modulus_avg(
        zeta_hsp, *(v for i, v in enumerate(args) if i % 3 != 0)
    )
    mu_hsm = _hsw_shear_modulus_avg(
        zeta_hsm, *(v for i, v in enumerate(args) if i % 3 != 0)
    )
    to_shp = check_broadcastable(
        **{
            "k_hsp": k_hsp,
            "k_hsm": k_hsm,
            "mu_hsp": mu_hsm,
            "mu_hsm": mu_hsm,
        }
    )

    return tuple(np.broadcast_to(ar, to_shp) for ar in (k_hsp, k_hsm, mu_hsp, mu_hsm))


def hsw_avg(
    bulk_mod: NDArrayOrFloat,
    shear_mod: NDArrayOrFloat,
    frac: NDArrayOrFloat,
    *argv: NDArrayOrFloat,
) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
    """Hashin-Shtrikman-Walpole average moduli for N materials.

    Args:
        bulk_mod: first component bulk modulus
        shear_mod: first component shear modulus
        frac: first component volume fraction
        *argv: additional triplets of bulk_mod, shear_mod and frac for additional components

    Returns:
        K_hs_avg, Mu_hs_avg
    """
    k_hsp, k_hsm, mu_hsp, mu_hsm = hsw_bounds(bulk_mod, shear_mod, frac, *argv)
    return 0.5 * (k_hsp + k_hsm), 0.5 * (mu_hsp + mu_hsm)
