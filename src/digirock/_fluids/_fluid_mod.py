"""Fluid models to simplify generation of fluid properties.

"""
from typing import List, Dict, Tuple

# pylint: disable=invalid-name,no-value-for-parameter
import numpy as np

from .._exceptions import PrototypeError, WorkflowError
from ..utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from ..utils.ecl import EclStandardConditions, EclUnitMap, EclUnitScaler
from ..utils._decorators import mutually_exclusive, check_props, broadcastable

from ..fluids import bw92

from ._fluid import Fluid
from ._oil import BaseOil


class FluidModel:
    """Class for defining and building fluid models.

    Fluids are definied by adding component fluids to the model. Fluid properties can then be
    queried by submitting conditions.

    Attributes:
        name (str): Name of Fluid Model
        components (dict): component fluid properties
        ncomp (int): number of component fluids
    """

    def __init__(self, name=None, mixing_model="woods"):
        self.name = name
        self.components = dict()
        self._bdmcol.append("components")
        self.ncomp = 0
        self.mixing_model = mixing_model

    def add_component(self, name, fluid):
        """Add a fluid component to the model.

        Args:
            name (str): Name of fluid for key reference in other methods.
            fluid (etlpy.pem.Fluid): Derivative class of Fluid base class.
        """
        if not isinstance(fluid, Fluid):
            raise ValueError(f"fluid should be of type {Fluid} got {type(fluid)}")
        self.components[name] = fluid

    # def print fluids

    def _check_kwargs(self, vol_frac_tol=1e-3, **kwargs):
        if not kwargs:
            raise ValueError(f"Specify at least two fluids to mix.")

        lengths = []
        nonekey = None
        for key in kwargs:
            if key not in self.components.keys():
                raise ValueError(f"Unknown fluid keyword {key}")
            if kwargs[key] is None and nonekey is None:
                nonekey = key
            elif kwargs[key] is None and nonekey is not None:
                raise ValueError("Only one fluid component can be the complement")

            if not isinstance(kwargs[key], np.ndarray) and nonekey != key:
                lengths.append(1)
            elif nonekey != key:
                lengths.append(kwargs[key].size)

        n = len(lengths)
        if np.all(np.array(lengths) == lengths[0]):
            frac_test = np.zeros((lengths[0], n))
        else:
            raise ValueError(
                f"Input volume fractions must be the same size got {lengths}"
            )

        i = 0
        for key in kwargs:
            if key != nonekey:
                frac_test[:, i] = kwargs[key]
                i = i + 1

        if nonekey is not None:
            if np.all(frac_test.sum(axis=1) <= (1.0 + vol_frac_tol)):
                kwargs[nonekey] = 1 - frac_test.sum(axis=1)
            else:
                raise ValueError(
                    f"Input volume fractions sum to greater than 1"
                    + f" tolerance is {vol_frac_tol} and max sum was {np.max(frac_test.sum(axis=1))}"
                    + f" for keys {list(kwargs.keys())}"
                )
        else:
            if not np.allclose(frac_test.sum(axis=1), 1.0):
                raise ValueError(
                    f"Input volume fractions must sum to 1 if no complement."
                )

        return kwargs

    def _strip_oil_kwargs(self, kwargs):
        if "bo" in kwargs:
            bo = kwargs.pop("bo")
        else:
            bo = None

        if "rs" in kwargs:
            rs = kwargs.pop("rs")
        else:
            rs = None
        return kwargs, bo, rs

    def density(self, temp, pres, vol_frac_tol=1e-3, **kwargs):
        """Return the density of the mixed fluid.

        The arguments passed to this function are the volume fractions of each fluid name to mix.

        Volume fractions should sum to 1, pass a single fluid with value as None to set it
        as the complement.

        Args:
            kwargs: Volume fraction array.
        """
        # TODO: check temp and pres are good dims with kwargs

        kwargs, bo, rs = self._strip_oil_kwargs(kwargs)

        kwargs = self._check_kwargs(vol_frac_tol=vol_frac_tol, **kwargs)
        args = []
        for key, frac in kwargs.items():

            if isinstance(self.components[key], BaseOil):
                args = args + [
                    self.components[key].density(temp, pres, rs=rs, fvf=bo),
                    frac,
                ]
            else:
                args = args + [self.components[key].density(temp, pres), frac]
        return bw92.mixed_density(*args)

    def modulus(self, temp, pres, vol_frac_tol=1e-3, **kwargs):
        """Return the modulus of the mixed fluid.

        The kw arguments pass to this function are the volume fractions of each fluid name to mix.

        Volume frations should sum to 1, pass a single fluid with value as None to set it as the
        complement.

        Args:
            temp: Temperature for sameple point/s.
            pres: Pressure for sample point/s.
            kwargs: Volume fraction array for fluid kw.

        Returns:
           : The modulus of the mixed fluid for temp and pres points.
        """
        kwargs, bo, rs = self._strip_oil_kwargs(kwargs)
        kwargs = self._check_kwargs(vol_frac_tol=vol_frac_tol, **kwargs)
        args = []
        for key, frac in kwargs.items():
            if isinstance(self.components[key], BaseOil):
                args = args + [
                    self.components[key].density(temp, pres, rs=rs, fvf=bo),
                    frac,
                ]
            else:
                args = args + [self.components[key].density(temp, pres), frac]
        if self.mixing_model == "woods":
            return bw92.mixed_bulkmod(*args)

    def get_summary(self) -> dict:
        return {comp: self.components[comp].get_summary() for comp in self.components}
