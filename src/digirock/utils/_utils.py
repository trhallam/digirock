from typing import Sequence
import numpy as np
import xarray as xr

from ..typing import NDArrayOrFloat


# def _check_kwargs_vfrac(**kwargs):
#     """Check that kwargs sum to volume fraction of one.

#     Raises:
#         ValueError: [description]
#         ValueError: [description]
#         ValueError: [description]
#         ValueError: [description]

#     Returns:
#         [type]: [description]
#     """
#     lengths = []
#     nonekey = None
#     for key in kwargs:
#         if kwargs[key] is None and nonekey is None:
#             nonekey = key
#         elif kwargs[key] is None and nonekey is not None:
#             raise ValueError("Only one component can be the complement")

#         if not isinstance(kwargs[key], np.ndarray) and nonekey != key:
#             lengths.append(1)
#         elif nonekey != key:
#             lengths.append(kwargs[key].size)

#     n = len(lengths)
#     if np.all(np.array(lengths) == lengths[0]):
#         frac_test = np.zeros((lengths[0], n))
#     else:
#         raise ValueError(f"Input volume fractions must be the same size got {lengths}")

#     i = 0
#     for key in kwargs:
#         if key != nonekey:
#             frac_test[:, i] = kwargs[key]
#             i = i + 1

#     if nonekey is not None:
#         if np.all(frac_test.sum(axis=1) <= 1.0):
#             kwargs[nonekey] = 1 - frac_test.sum(axis=1)
#         else:
#             raise ValueError(f"Input volume fractions sum to greater than 1")
#     else:
#         if not np.allclose(frac_test.sum(axis=1), 1.0):
#             raise ValueError(f"Input volume fractions must sum to 1 if no complement.")

#     return kwargs


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


def _process_vfrac(
    *argv: NDArrayOrFloat, i=1, tol: float = 1e-6
) -> Sequence[NDArrayOrFloat]:
    """Process an arbitrary number of components to return broadcastable arrays for volume based mixing.

    Can take an arbitrary number of components and volume fractions.
    Volume fractions must sum to one for each sample.
    If the number of arguments is odd, the final frac is assumed to be the complement of all other fracs to sum to 1.

    Inputs must be [broadcastable][https://numpy.org/doc/stable/user/basics.broadcasting.html].

    When i==1:
    `argv` as the form `(component_1, vfrac_1, component_2, vfrac2, ...)` or to use the complement for the final
    component `(component_1, vfrac_1, component_2)`

    Args:
        argv: Alternating component and volume fraction inputs
        tol: The volume sum absolute tolerance

    Returns:
        argv in processable state

    Raises:
        ValueError: Volume fractions != 1 +- tol
    """
    interval = i + 1
    try:
        assert (len(argv) - 1) % interval - (interval - 2) >= 0
    except AssertionError:
        raise ValueError(
            f"Length of arguments was l={len(argv)}, expected n*{interval} or n*{interval}-1"
        )
    to_shp = check_broadcastable(**{f"argv{i}": arg for i, arg in enumerate(argv)})
    # check arguments and find complement if necessary
    sum_vol = 0
    for arg in argv[i::interval]:
        sum_vol = sum_vol + np.array(arg)

    # find complement if necessary
    if len(argv) % interval != 0:
        fvol = np.clip(1 - sum_vol, 0, 1)
        sum_vol = sum_vol + fvol
        argv = argv + (np.clip(fvol, 0, 1),)

    # check volume fractions sum to one
    if not np.allclose(1, sum_vol, atol=tol):
        raise ValueError(
            f"Volume fractions do not sum to 1, got sum_vol min: {sum_vol.min()} max: {sum_vol.max()}"
        )

    return argv


def safe_divide(a: NDArrayOrFloat, b: NDArrayOrFloat) -> NDArrayOrFloat:
    """Helper function to avoid divide by zero in arrays and floats.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        a/b replace div0 by 0
    """
    bc_shp = check_broadcastable(a=a, b=b)
    return np.divide(a, b, out=np.zeros(bc_shp), where=b != 0.0)


def nan_divide(a: NDArrayOrFloat, b: NDArrayOrFloat) -> NDArrayOrFloat:
    """Helper function to avoid divide by zero in arrays and floats.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        a/b replace div0 by np.nan
    """
    bc_shp = check_broadcastable(a=a, b=b)
    return np.divide(a, b, out=np.full(bc_shp, np.nan), where=b != 0.0)


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


def create_orthogonal_props(**coords: NDArrayOrFloat):
    """Creates a set of orthogonal arrays and returns them as xr.DataArrays in
    a props dictionary.

    Args:
        coords: Linear arrays of coordinates, e.g. poro, VSH

    Returns:
        the coordinate xr.Dataset, a dictionary of props xr.DataArray for each of coords broadcast against each other

    Example:
    ```
    ds, props = create_orthogonal_props(VSH=np.linspace(0, 0.15, 2), ncontacts=np.arange(10, 21, 10), poro=np.r_[0.1, 0.2])
    print(ds)
    <xarray.Dataset>
    Dimensions:    (VSH: 2, ncontacts: 2, poro: 2)
    Coordinates:
    * VSH        (VSH) float64 0.0 0.15
    * ncontacts  (ncontacts) int64 10 20
    * poro       (poro) float64 0.1 0.2
    Data variables:
    *empty*

    print(props)
    {'VSH': <xarray.DataArray 'bc_VSH' (VSH: 2, ncontacts: 2, poro: 2)>
    array([[[0.  , 0.  ],
        [0.  , 0.  ]],

        [[0.15, 0.15],
        [0.15, 0.15]]])
    Coordinates:
    * VSH        (VSH) float64 0.0 0.15
    * ncontacts  (ncontacts) int64 10 20
    * poro       (poro) float64 0.1 0.2, 'ncontacts': <xarray.DataArray 'bc_ncontacts' (VSH: 2, ncontacts: 2, poro: 2)>
    array([[[10, 10],
        [20, 20]],

        [[10, 10],
        [20, 20]]])
    Coordinates:
    * VSH        (VSH) float64 0.0 0.15
    * ncontacts  (ncontacts) int64 10 20
    * poro       (poro) float64 0.1 0.2, 'poro': <xarray.DataArray 'bc_poro' (VSH: 2, ncontacts: 2, poro: 2)>
    array([[[0.1, 0.2],
        [0.1, 0.2]],

        [[0.1, 0.2],
        [0.1, 0.2]]])
    Coordinates:
    * VSH        (VSH) float64 0.0 0.15
    * ncontacts  (ncontacts) int64 10 20
    * poro       (poro) float64 0.1 0.2}
    ```
    """
    ds = xr.Dataset(coords={key: (key, val) for key, val in coords.items()})
    # for key in coords:
    bc = xr.broadcast(*tuple(ds[key] for key in coords))
    bc = xr.Dataset(data_vars={f"bc_{key}": bcv for key, bcv in zip(coords, bc)})
    return ds, {key: bc[f"bc_{key}"].values for key in coords}
