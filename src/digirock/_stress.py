# import xarray as xr
import numpy as np

from ._exceptions import WorkflowError
from addict import Dict as AttributeDict

from .models._mod import dryframe_dpres


class StressModel:
    """Build a stress model for a rock.

    Attributes:
        sp (dict): Stress sensitivity parameters
        grad (float): Stress overburden gradient
        ref_pres (float): Pressure (MPa) at ref_depth
        ref_depth (float): Reference depth (mTVDSS) for ref_pres.
    """

    _has_stress_sens = False
    _has_ob_stress = False
    # sp = AttributeDict(
    #   {"ek": None, "pk": None, "infk": None, "eg": None, "pg": None, "infg": None}
    # )
    grad = None
    ref_pres = None
    ref_depth = None

    def __init__(self, name=None):
        self.name = name
        save_header = "etlpy StressModel class instance save file"
        super().__init__(save_header)

    def set_sensitivity(self, ek, pk, eg, pg):
        """Stress sensitivity parameters.
        """
        self.sp = AttributeDict({"ek": ek, "pk": pk, "eg": eg, "pg": pg,})
        self._has_stress_sens = True

    def set_stress_env(self, grad, ref_pres, ref_depth):
        """Background stress gradient - can by hydrostatic for example.

        Sv = grad*(depth-ref_depth) + ref_pres

        Args:
            grad (float): The gradient of the background stress in MPa/m
            ref_pres (float): The intercept of the stress graident trend i.e. at ref_depth (MPa)
            ref_depth (float): Depth at reference pressure (TVDSS)
        """
        self.grad = grad
        self.ref_pres = ref_pres
        self.ref_depth = ref_depth
        self._has_ob_stress = True
        self._sv_func = self._vertical_stress_from_env

    def _vertical_stress_from_env(self, depth):
        """Calculate vertical stress from linear gradient terms.

        Args:
            depth (array-like): Depth to calculate at.

        Returns:
            (array-like): Vertical stress (MPa)
        """
        if not self._has_ob_stress:
            raise WorkflowError(
                "vertical_stress", "overburden stress must be set using set_stress_env"
            )
        return self.grad * (depth - self.ref_depth) + self.ref_pres


    def set_vertical_stress_func(self, func):
        """Set a custom stress function for the overburden.

        Depths are positive down.

        Args:
            func (function): A function which takes a single argument (tvdss) and
                returns an array of the same size for vertical pressure in MPa

        """
        test = np.arange(0, 4000)
        assert func(test).size == test.size
        assert np.all(func(test) >= 0)
        self._sv_func = func


    def vertical_stress(self, depth):
        return self._sv_func(depth)


    def effective_stress(self, depth, res_pres):
        """Effective stress by taking difference of vertical stress and reservoir pressure.

        Args:
            depth (array-like): depth of calculation (m TVDSS)
            res_pres (array-like): reservoir pressure (MPa)

        Returns:
            (array-like): effective stress (MPa)
        """
        return self.vertical_stress(depth) - res_pres

    def stress_dryframe_moduli(self, depth, pres1, pres2, mod_dry, component="bulk"):
        """Calculate the dryframe stress sensitive moduli

        Args:
            depth ([type]): [description]
            pres1 ([type]): [description]
            pres2 (array-like):
            mod_dry (array-like): Dryframe modulus at pressure 1.
        """
        if not self._has_stress_sens:
            raise WorkflowError(
                "dryframe_moduli",
                "The stress sensitivity parameters have not been set.",
            )

        if component not in ["bulk", "shear"]:
            raise ValueError(f"Unknown component {component}")

        peffi = self.effective_stress(depth, pres1)
        peff = self.effective_stress(depth, pres2)

        if component == "bulk":
            return dryframe_dpres(mod_dry, peffi, peff, self.sp.ek, self.sp.pk)
        if component == "shear":
            return dryframe_dpres(mod_dry, peffi, peff, self.sp.eg, self.sp.pg)
