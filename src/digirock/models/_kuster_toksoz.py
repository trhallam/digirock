"""Kuster-Toksoz Model for effective moduli for rocks.

Refs:
    Mavko et. al. (2009) The Rock Physics Handbook, Cambridge Press
"""

# pylint: disable=invalid-name

from numpy import power, pi
from numpy import all as npall


def _kuster_toksoz_beta(k, mu):
    """Inputs must be same shape

    Args:
        k (array-like): Bulk Modulus
        mu (array-like): Shear Modulus

    Returns:
        (array-like): beta term
    """
    return (mu * (3 * k + mu)) / (3 * k + 4 * mu)


def _kuster_toksoz_gamma(k, mu):
    """Inputs must be same shape

    Args:
        k (array-like): Bulk Modulus
        mu (array-like): Shear Modulus

    Returns:
        (array-like): gamma term
    """
    return (mu * (3 * k + mu)) / (3 * k + 7 * mu)


def _kuster_toksoz_eta(k, mu):
    """Inputs must be same shape

    Args:
        k (array-like): Bulk Modulus
        mu (array-like): Shear Modulus

    Returns:
        (array-like): eta term
    """
    return (mu * (9 * k + 8 * mu)) / (6 * (k + 2 * mu))


def _kuster_toksoz_spheres(k_m, mu_m, k_i, mu_i):
    """Coefficients for Spherical sphaped inclusions.
    Inputs must be same shape.

    Args:
        k_m (array-like): Bulk modulus of material to get inclusions.
        mu_m (array-like): Shear modulus of material to get inclusions.
        k_i (array-like): Bulk modulus of material to be added as inclusions.
        mu_i (array-like): Shear modulus of material to be added inclusions.

    Returns:
        pmi, qmi (array-like): P and Q KT coefficients
    """
    eta_m = _kuster_toksoz_eta(k_m, mu_m)
    return (
        (k_m + 4 * mu_m / 3) / (k_i + 4 * mu_i / 3),
        (mu_m + eta_m) / (mu_i + eta_m),
    )


def _kuster_toksoz_needles(k_m, mu_m, k_i, mu_i):
    """Coefficients for Needle sphaped inclusions.
    Inputs must be same shape.

    Args:
        k_m (array-like): Bulk modulus of material to get inclusions.
        mu_m (array-like): Shear modulus of material to get inclusions.
        k_i (array-like): Bulk modulus of material to be added as inclusions.
        mu_i (array-like): Shear modulus of material to be added inclusions.

    Returns:
        pmi, qmi (array-like): P and Q KT coefficients
    """
    gamma_m = _kuster_toksoz_gamma(k_m, mu_m)
    return (
        (k_m + mu_m + mu_i / 3) / (k_i + mu_m + mu_i / 3),
        0.2
        * (
            (4 * mu_m) / (mu_m + mu_i)
            + 2 * (mu_m + gamma_m) / (mu_i + gamma_m)
            + (k_i + 4 * mu_m / 3) / (k_i + mu_m + mu_i / 3)
        ),
    )


def _kuster_toksoz_disks(k_m, mu_m, k_i, mu_i):
    """Coefficients for Disk sphaped inclusions. (Zero Thickness Cracks)
    Inputs must be same shape.

    Args:
        k_m (array-like): Bulk modulus of material to get inclusions.
        mu_m (array-like): Shear modulus of material to get inclusions.
        k_i (array-like): Bulk modulus of material to be added as inclusions.
        mu_i (array-like): Shear modulus of material to be added inclusions.

    Returns:
        pmi, qmi (array-like): P and Q KT coefficients
    """
    eta_i = _kuster_toksoz_eta(k_i, mu_i)
    return (
        (k_m + 4 * mu_i / 3) / (k_i + 4 * mu_i / 3),
        (mu_m + eta_i) / (mu_i + eta_i),
    )


def _kuster_toksoz_cracks(k_m, mu_m, k_i, mu_i, alpha):
    """Coefficients for crack sphaped inclusions.
    Inputs must be same shape.

    Args:
        k_m (array-like): Bulk modulus of material to get inclusions.
        mu_m (array-like): Shear modulus of material to get inclusions.
        k_i (array-like): Bulk modulus of material to be added as inclusions.
        mu_i (array-like): Shear modulus of material to be added inclusions.
        alpha (float): Aspect ratio of cracks (in range [0:1])

    Returns:
        pmi, qmi (array-like): P and Q KT coefficients
    """
    if not npall(0 <= alpha <= 1):
        raise ValueError("alpha must be ratio in range {0:1}")
    beta_m = _kuster_toksoz_beta(k_m, mu_m)
    return (
        (k_m + 4 * mu_i / 3) / (k_i + 4 * mu_i / 3 + pi * alpha * beta_m),
        0.2
        * (
            1
            + (8 * mu_m) / (4 * mu_i + pi * alpha * (mu_m + 2 * beta_m))
            + 2
            * (k_i + 2 / 3 * (mu_i + mu_m))
            / (k_i + 4 * mu_i / 3 + pi * alpha * beta_m)
        ),
    )


def kuster_toksoz_moduli(
    k1, mu1, k2, mu2, frac2, inclusion_shape="spheres", alpha=None
):
    """Kuster-Toksoz Moduli for an inclusion to a material. Best used for low-porosity materials.

    To add multiple inclusions to a model use this function recursively substituting the output for
    k1 and mu1 after the first pass.

    Inclusions are added randomly (iso-tropic).
    Assumes the material is learn and elastic, is limited to dilute concentrations of inclusions and
    idealised ellipsoidal inclusion shapes.

    Args:
        k1 (array-like): Material bulk moduli
        mu1 (array-like): Material shear moduli
        k2 (array-like): Inclusion bulk moduli
        mu2 (array-like): Inclusion shear moduli
        frac2 (array-like): The volume fraction of the inclusion to be added.
        inclusion_shape (str, Optional): The shape of the inclusion. Defaults to 'spheres'. One of
            ['spheres', 'needles', 'disks', 'cracks'].
        alpha (float, Optional): Required if inclusion_shape='cracks'. The aspect ratio of the
            cracks.

    """
    if inclusion_shape == "spheres":
        pmi, qmi = _kuster_toksoz_spheres(k1, mu1, k2, mu2)
    elif inclusion_shape == "needles":
        pmi, qmi = _kuster_toksoz_needles(k1, mu1, k2, mu2)
    elif inclusion_shape == "disks":
        pmi, qmi = _kuster_toksoz_disks(k1, mu1, k2, mu2)
    elif inclusion_shape == "cracks" and isinstance(alpha, float):
        pmi, qmi = _kuster_toksoz_cracks(k1, mu1, k2, mu2, alpha)
    else:
        raise ValueError(
            "Unknown inclusions_shape or alpha must be specified as float for cracks."
        )

    eta1 = _kuster_toksoz_eta(k1, mu1)
    k_a = frac2 * (k2 - k1) * pmi
    mu_a = frac2 * (mu2 - mu1) * qmi
    return (
        (4 / 3 * mu1 * (k_a + k1) + power(k1, 2)) / (k1 + 4 * mu1 / 3 - k_a),
        (eta1 * (mu_a + mu1) + power(mu1, 2)) / (mu1 + eta1 - mu_a),
    )
