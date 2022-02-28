"""Classes for handling Stress fields.
"""
from inspect import getargspec
from typing import Dict, Iterable, List

from ._base import Element
from ._exceptions import PrototypeError, WorkflowError
from .utils._decorators import check_props
from .typing import NDArrayOrFloat


class StressModel(Element):
    """Base Class for defining stress fields, all new stress fields should be based upon this class.

    Attributes:
        name (str): name of the field
    """

    def __init__(self, name: str = None, keys: List[str] = None):
        if keys is None:
            keys = ["depth", "pres"]
        super().__init__(name, keys)

    def vertical_stress(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Returns the vertical stress $S_v$ for the class.

        Args:
            props: A dictionary of properties required
            kwargs: ignored
        """
        raise PrototypeError(self.__class__.__name__, "vertical_stress")

    def effective_stress(self, props: Dict[str, NDArrayOrFloat], **kwargs):
        """Returns the effective stress $S_e$ for the class at a given depth $(z)$ and for a
        particular formation pressure $p_f$.

        $S_e = S_v(d) - p_f$

        Args:
            props: A dictionary of properties; requires `depth` (m) and `pres` (MPa)
            kwargs: ignored

        Returns:
            effective stress (MPa)
        """
        return self.vertical_stress(props) - props["pres"]

    def get_summary(self) -> dict:
        """Return a dictionary containing a summary of the fluid.

        Returns:
            Summary of properties.
        """
        return super().get_summary()


class FStressModel(StressModel):
    """Build a stress model for a rock using a user defined function.

    The function `func` passed to the constructor should be of the form

    ```
    def stress_func(props: Dict[str, NDArrayOrFloat] = None, **kwargs) -> NDArrayOrFloat:
        return props.
    ```

    where props contains constants or properties for each calculation point and returns an NDArray of the same
    size or a constant. The keys required by stress_func should be given to the `keys` argument of the constructor
    to enable property_checks at runtime.

    Attributes:
        stress_func (callable): Stress function
    """

    def __init__(self, func, name: str = None, keys: List[str] = None):
        """

        Args:
            func: the vertical stress function
            name: name of the model
            keys: A list of property keys that this function requires, e.g. `'depth'`
        """
        super().__init__(name=name, keys=keys)
        if keys is None:
            keys = []

        # check func should have form func(props, **kwargs)
        spec = getargspec(func)
        try:
            assert len(spec.args) == 1
            assert spec.keywords == "kwargs"
        except AssertionError:
            raise ValueError("func must have the form `func(props, **kwargs)`")

        # TODO: get update chekc_props
        # self.stress_func = check_props(*tuple(keys))(func)
        self.stress_func = func

    def vertical_stress(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns the vertical stress using user defined function `vertical_stress`.

        Returns:
            vertical stress (MPa)
        """
        return self.stress_func(props, **kwargs)

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update(
            {
                "func": self.stress_func.__name__,
                "func_argspec": getargspec(self.stress_func),
            }
        )
        return summary


class LGStressModel(StressModel):
    """Build a stress model for a rock using a linear gradient (LG).

    Attributes:
        grad (float): Stress overburden gradient
        ref_pres (float): Pressure (MPa) at ref_depth
        ref_depth (float): Reference depth (mTVDSS) for ref_pres.
    """

    grad = None
    ref_pres = None
    ref_depth = None

    def __init__(self, grad: float, ref_pres: float, ref_depth: float = 0, name=None):
        """Background stress set using a linear gradient
        ```
        Sv = grad*(depth-ref_depth) + ref_pres
        ```

        Args:
            grad: The gradient of the background stress in MPa/m
            ref_pres: The intercept of the stress gradient trend i.e. at `ref_depth` (MPa)
            ref_depth: Depth at reference pressure (mTVDSS)
        """
        self.grad = grad
        self.ref_pres = ref_pres
        self.ref_depth = ref_depth
        super().__init__(name, keys=["depth", "pres"])

    # @check_props("depth")
    def vertical_stress(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        return self.grad * (props["depth"] - self.ref_depth) + self.ref_pres

    def get_summary(self) -> dict:
        summary = super().get_summary()
        summary.update(
            {"grad": self.grad, "ref_pres": self.ref_pres, "ref_depth": self.ref_depth}
        )
        return summary
