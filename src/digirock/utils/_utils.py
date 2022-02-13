import numpy as np

from digirock.utils.types import NDArrayOrFloat


def _check_kwargs_vfrac(**kwargs):
    """Check that kwargs sum to volume fraction of one.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    lengths = []
    nonekey = None
    for key in kwargs:
        if kwargs[key] is None and nonekey is None:
            nonekey = key
        elif kwargs[key] is None and nonekey is not None:
            raise ValueError("Only one component can be the complement")

        if not isinstance(kwargs[key], np.ndarray) and nonekey != key:
            lengths.append(1)
        elif nonekey != key:
            lengths.append(kwargs[key].size)

    n = len(lengths)
    if np.all(np.array(lengths) == lengths[0]):
        frac_test = np.zeros((lengths[0], n))
    else:
        raise ValueError(f"Input volume fractions must be the same size got {lengths}")

    i = 0
    for key in kwargs:
        if key != nonekey:
            frac_test[:, i] = kwargs[key]
            i = i + 1

    if nonekey is not None:
        if np.all(frac_test.sum(axis=1) <= 1.0):
            kwargs[nonekey] = 1 - frac_test.sum(axis=1)
        else:
            raise ValueError(f"Input volume fractions sum to greater than 1")
    else:
        if not np.allclose(frac_test.sum(axis=1), 1.0):
            raise ValueError(f"Input volume fractions must sum to 1 if no complement.")

    return kwargs


def check_broadcastable(**kwargs: NDArrayOrFloat) -> tuple:
    """Check that kwargs can be numpy broadcast against each other for matrix
    operations.

    Args:
        kwargs: Keywords to check with float of array-like values.

    Returns
        Shapes of broadcast keywords

    Raises:
        ValueError if not broadcastable
    """
    check_argshapes = [np.atleast_1d(arg).shape for arg in kwargs.values()]

    try:
        shapes = np.broadcast_shapes(*tuple(check_argshapes))
    except:
        msg = f"Cannot broadcast shapes: " + ", ".join(
            [f"{name}:{shp}" for name, shp in zip(kwargs, check_argshapes)]
        )
        raise ValueError(msg)

    return shapes


def safe_divide(a, b):
    """Helper function to avoid divide by zero in many areas.
    Args:
        a (array-like): Numerator
        b (array-like): Deominator
    Returns:
        a/b (array-like): Replace div0 by 0
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    shp = a.shape
    a = np.squeeze(a)
    b = np.squeeze(b)
    shp = a.shape if a.size != 1 else b.shape
    return np.divide(a, b, out=np.zeros(shp), where=b != 0.0)


def ndim_index_list(n):
    """Creates the list for indexing input data based upon dimensions in list n.

    As input takes a list n = [n0,n1,n2] of the length of each axis in order of
    slowest to fastest cycling. Such that:
    n0 is numbered 1,1,1, 1,1,1, 1,1,1; 2,2,2, 2,2,2, 2,2,2; 3,3,3, 3,3,3, 3,3,3;
    n1 is numbered 1,1,1, 2,2,2, 3,3,3; 1,1,1, 2,2,2, 3,3,3; 1,1,1, 2,2,2, 3,3,3;
    n2 is numbered 1,2,3, 1,2,3, 1,2,3; 1,2,3, 1,2,3, 1,2,3; 1,2,3, 1,2,3, 1,2,3;

    Args:
        n (list): a list of intergers giving the dimensions of the axes.

    Note: axes not required are given values n = 1
          indexing starts from 1, Zero based indexing is available by subtracting
          1 from all values.
    """
    if isinstance(n, list):
        ind = list()
        ind.append(np.arange(1, n[0] + 1).repeat(np.product(n[1:])))
        for i in range(1, len(n) - 1):
            ni = i + 1
            t = np.arange(1, n[i] + 1)
            t = t.repeat(np.product(n[ni:]))
            t = np.tile(t, np.product(n[:i]))
            ind.append(t)
        ind.append(np.tile(np.arange(1, n[-1] + 1), np.product(n[:-1])))
        return ind
    else:
        raise ValueError("n must be of type list containing integer values")
