"""RockFrame models to simplify generation of digitial rock frames.

"""

# pylint: disable=invalid-name,no-value-for-parameter

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

from ..utils.file import read_eclipsekw_3dtable, read_eclipsekw_2dtable
from .._exceptions import PrototypeError, WorkflowError
from ..utils._decorators import mutually_exclusive
from ..utils.ecl import EclUnitScaler
from ..utils._utils import _check_kwargs_vfrac

from ..models import _mod
from .._stress import StressModel
from ..models._cemented_sand import dryframe_cemented_sand


class PoroAdjModel:
    """Base Class for defining porosity adjustment models.

    Attributes:
        name (str): name of the model
    """

    def __init__(self, name=None):
        save_header = "etlpy porosity adjustment model class"
        super().__init__(save_header)
        self.name = name

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform input porosity using model"""
        raise PrototypeError(self.__class__.__name__, "porosity model")


class DefaultPoroAdjModel:
    """Default porosity adjustment case

    Attributes:
        name (str): name of the model
    """

    def __init__(self, name=None):
        save_header = "etlpy porosity adjustment model class"
        super().__init__(save_header)
        self.name = name

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform input porosity using model"""
        return 1 - 2.8 * porosity


class NurCriticalPoro(PoroAdjModel):
    """Nur's Critical porosity adjustment."""

    def __init__(self, name=None, critical_poro=None):
        super().__init__(name)
        self.set_critical_porosity(critical_poro)

    def set_critical_porosity(self, crit_por):
        """Critical porosity parameters for Nur's critical porosity model.

        Args:
            crit_por (float): Critical porosity inflection point in Por vs Vp (0 < crit_por < 1)
        """
        self.crit_porosity = crit_por

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform the input porosity using Nur's critical porosity model.

        component is ignored for Nur's CPor model.
        """
        if self.crit_porosity is None:
            raise WorkflowError(
                "porosity_adjustment",
                "Critical porosity must be set to use Nur critical model",
            )
        if not (0 <= self.crit_porosity <= 1):
            raise ValueError("Critical posority (cp) should be 0 <= cp <= 1")

        return np.where(
            porosity >= self.crit_porosity, 0.0, 1 - porosity / self.crit_porosity
        )


class LeeConsolodationPoro(PoroAdjModel):
    """Lee/Pride 2005 Consolidation parameter porosity adjustment."""

    def __init__(self, name=None, cons_alpha=None):
        super().__init__(name)
        if cons_alpha is not None:
            self.set_consolidation(cons_alpha)
        else:
            self.cons_alpha = None

    def set_consolidation(self, alpha, gamma=None):
        """Set consolidation parameters after Lee (2005).

        dry frame modulus will be modified after
            kdry = k0 * (1 - phi)/(1 - alpha*phi)
            mudry = mu0 * (1 - phi)/(1 - alpha*gamma*phi)

        If gamma is None:
            gamma = (1 + 2*alpha) / (1 + alpha)

        Args:
            alpha (): [description]
        """
        self.cons_alpha = alpha
        if gamma is None:
            self.cons_gamma = (1 + 2 * alpha) / (1 + alpha)
        else:
            self.cons_gamma = gamma

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform the input porosity using Lee's consolidation porosity model."""
        if self.cons_alpha is None:
            raise WorkflowError(
                "porosity_adjustment",
                " consolidation parameter must be set",
            )

        if component == "bulk":
            return (1 - porosity) / (1 + self.cons_alpha * porosity)
        elif component == "shear":
            return (1 - porosity) / (1 + self.cons_gamma * self.cons_alpha * porosity)
        else:
            raise ValueError(f"Unknown component type {component}")


class WoodsideCementPoro(PoroAdjModel):
    """Woodside's cemented sand model"""

    def __init__(self, name=None):
        super().__init__(name)
        self.cement_index = dict()

    def set_cement_index(self, cement_index, component="bulk"):
        """Cement Index for Woodside Modulus Adjustment

        Args:
            cement_index ([type]): [description]
            component (str, optional): [description]. Defaults to "bulk".
        """
        self.cement_index[component] = cement_index

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform the input porosity using Lee's consolidation porosity model."""
        try:
            ci = self.cement_index[component]
        except KeyError:
            raise WorkflowError(
                "porosity_adjustment",
                f" cement index for component ({component}) must be set",
            )

        return 1 - porosity / (porosity + ci)


class ETLPConsolodationPoro(PoroAdjModel):

    """ETLP original sim2seis Consolidation parameter porosity adjustment."""

    def __init__(self, name=None):
        raise NotImplementedError("This model is not finished yet")
        super().__init__(name)
        self.etlp_cons = None

    def set_etlp_consolidation(self, **kwargs):
        """ETLP Consolidation Parameters c1, c2, c3, c4, phi & depth must be set.

        cX should be replaced by the mineral key passed as a vfrac in transform.

        Args:
            alphas (dict): [description]
        """
        if len(kwargs) > 0:
            self.etlp_cons = kwargs

    def transform(self, porosity, component="bulk", **min_kwargs):
        """Transform the input porosity using ETLP consolidation porosity
        model.

        Notes:
            True origin is not well documents but is coded up in the original matlab sim2seis.
        """
        if self.etlp_cons is None:
            raise WorkflowError(
                "ETLP Consolodation Parameter",
                "ETLP Consolodation Parameters must be set",
            )
        try:
            por_cp = self.etlp_cons["phi"]
        except KeyError:
            raise WorkflowError(
                "ETLP Consolodation Parameter",
                "The consolodation parameter for porosity phi is missing",
            )

        try:
            depth_cp = self.etlp_cons["depth"]
        except KeyError:
            raise WorkflowError(
                "ETLP Consolodation Parameter",
                "The consolodation parameter for depth is missing",
            )

        out = np.zeros_like(porosity)
        for mineral in min_kwargs:
            out = out + self.etlp_cons[mineral] * min_kwargs[mineral]
        # out = out + depth_cp * depth / 1000.0
        out = (1 - porosity) / (1 + out * porosity)
        return out
