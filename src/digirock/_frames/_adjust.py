"""Stress adjustment models. Usually this adjusts the porespace of your frame but can be
model specific.
"""
from typing import Dict, List, Tuple
from ..utils.types import NDArrayOrFloat
from ..utils._decorators import check_props
from .._base import BaseConsumerClass
from .._exceptions import WorkflowError, PrototypeError

from ..models import dryframe_dpres


class AdjustmentModel(BaseConsumerClass):
    """Base Class for adjusting properties, all new adjustment models should be based upon this class.

    Attributes:
        name (str): name of the adjustment
    """

    def __init__(self, name: str = None, keys: List[str] = None):
        if keys is None:
            keys = []
        BaseConsumerClass.__init__(self, name, keys)

    def _check_defined(self, from_func, var):
        if self.__getattribute__(var) is None:
            raise WorkflowError(from_func, f"The {var} attribute is not defined.")

    def adjust(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Adjust props to return props with adjustments applied.

        Args:
            props: A dictionary of properties required and to modify.
            kwargs: ignored
        """
        raise PrototypeError(self.__class__.__name__, "vertical_stress")

    def get_summary(self) -> dict:
        """Return a dictionary containing a summary of the fluid.

        Returns:
            Summary of properties.
        """
        return super().get_summary()


class MacBethStressAdjustment(AdjustmentModel):
    """Adjustment model for porosity based upon MacBeth's stress response model.

    The properties of `pres_e` and `pres_ei` must be calculated using a StressModel.
    peffi = self.effective_stress(depth, pres1)
    #     peff = self.effective_stress(depth, pres2)


    Attributes:
        e_k (float):
        p_k (float):
        e_mu (float):
        p_mu (float):
    """

    def __init__(
        self, e_k: float, p_k: float, e_mu: float, p_mu: float, name: str = None
    ):
        super().__init__(name=name, keys=["pres_e", "pres_ei"])
        self.e_k = e_k
        self.p_k = p_k
        self.e_mu = e_mu
        self.p_mu = p_mu

    @check_props("pres_e", "pres_ei")
    def adjust(
        self,
        props: Dict[str, NDArrayOrFloat],
        k_dry: str = "k_dry",
        mu_dry: str = "mu_dry",
        **kwargs,
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat]:
        """Dryframe porosity adjustment of bulk and shear moduli for stress.

        Args:
            props:
            k_dry:
            mu_dry:
            kwargs: ignored

        Returns
            updated props input
        """

        return (
            dryframe_dpres(
                props[k_dry], props["pres_ei"], props["pres_e"], self.e_k, self.p_k
            ),
            dryframe_dpres(
                props[mu_dry], props["pres_ei"], props["pres_e"], self.e_k, self.p_k
            ),
        )
