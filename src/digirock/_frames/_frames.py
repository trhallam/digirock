"""Classes for various RockFrames, new RockFrames should inherit `RockFrame` as a base class.
"""
from crypt import methods
from typing import Sequence, Dict, Type, List, Union
from .._base import Blend, Element
from ..typing import NDArrayOrFloat
from .. import models
from ..elastic import acoustic_vels, acoustic_velp


class RockFrame(Blend):
    """Base Class for defining rock frame models all new rock frames should be based upon this class.

    Attributes:
        name (str): name of the fluid
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods each element must have
        n_elements (int): The number of elements
    """

    def __init__(
        self,
        blend_keys: Sequence[str],
        elements: Sequence[Type[Element]],
        name=None,
    ):
        """

        Elements must implement `density()`, `vp()`, `vs()`, `shear_modulus()`, `bulk_modulus()`, `shear_modulus()`

        Args:
            blend_keys:
            elements:
            methods:
            name: Name of the rock frame

        """
        methods = ["density", "vp", "vs", "shear_modulus", "bulk_modulus"]
        super().__init__(blend_keys, elements, methods, name=name)

    def _process_props_get_method(
        self, props: Dict[str, NDArrayOrFloat], methods: Union[str, List[str]], **kwargs
    ) -> Sequence[NDArrayOrFloat]:
        """Process the props to find if all required keys are present for blending

        Uses self.blend_keys for check and get the result of method by passing props.

        Args:

        Returns:
            result for method from elements in order of `blend_keys`
        """
        missing = [key for key in self.blend_keys if key not in props]
        if len(missing) > 1:
            raise ValueError(
                f"Had {len(missing)} missing volume fractions, only 1 missing volume fraction allowed: please add to props {missing}"
            )
        has_keys = [key for key in self.blend_keys if key in props]

        args = []

        if isinstance(methods, str):
            methods = [methods]

        for key in has_keys:
            eli = self.blend_keys.index(key)
            args += [
                getattr(self._elements[eli], meth)(props, **kwargs) for meth in methods
            ] + [props[key]]

        if missing:
            eli = self.blend_keys.index(missing[0])
            args += [
                getattr(self._elements[eli], meth)(props, **kwargs) for meth in methods
            ]
        return tuple(args)

    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns density of RockFrame using volume fraction average, see [mixed_denisty][digirock.models._mod.mixed_density].

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            density (g/cc)
        """
        args = self._process_props_get_method(props, "density")
        return models.mixed_density(*args)
        # raise PrototypeError(self.__class__.__name__, "density")

    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns compression velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s).
        """
        density = self.density(props, **kwargs)
        bulk = self.bulk_modulus(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_velp(bulk, shear, density)

    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns shear velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            velocity (m/s).
        """
        density = self.density(props, **kwargs)
        shear = self.shear_modulus(props, **kwargs)
        return acoustic_vels(shear, density)

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns bulk modulus of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            bulk modulus (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "bulk_modulus")

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns shear modulus of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            shear modulus (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "shear_modulus")


class VRHAvg(RockFrame):
    """Voight-Reuss-Hill Averaging Class for combing mineral mixes based upon averaging.

    Attributes:
        name (str): name of the fluid
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods each element must have
        n_elements (int): The number of elements

    """

    def __init__(
        self,
        blend_keys: List[str],
        elements: List[Type[Element]],
        name=None,
    ):
        """

        Elements must implement `density()`, `vp()`, `vs()`, `shear_modulus()`, `bulk_modulus()`, `shear_modulus()`

        Args:
            blend_keys: props keys needed for blending calcs
            elements: elements to add to vrh averaging
            name: Name of the rock frame

        """
        super().__init__(
            blend_keys,
            elements,
            name=name,
        )

    def _vrh_avg_moduli(
        self,
        props: Dict[str, NDArrayOrFloat],
        component: str = "bulk_modulus",
        **kwargs,
    ) -> NDArrayOrFloat:
        """VRH avg moduli helper

        Args:
            props: A dictionary of properties required.
            component: One of ["bulk_modulus", "shear_modulus"]
            kwargs: passed to elements

        Returns:
            Voigt-Reuss-Hill Average modulus
        """
        args = self._process_props_get_method(props, component, **kwargs)

        return models.vrh_avg(*tuple(args))

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns bulk modulus of VRH average

        `props` must contain a volume fraction for each element

        Args:
            props: A dictionary of properties required
            kwargs: passed to elements

        Returns:
            Bulk modulus
        """
        return self._vrh_avg_moduli(props, component="bulk_modulus", **kwargs)

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns shear modulus of VRH average

        `props` must contain a volume fraction for each element

        Args:
            props: A dictionary of properties required
            kwargs: passed to elements

        Returns:
            Shear modulus
        """
        return self._vrh_avg_moduli(props, component="shear_modulus", **kwargs)


class HSWAvg(RockFrame):
    """Hashin-Striktman Averaging (can include a Fluid as element referenced to porosity.

    Attributes:
        name (str): name of the fluid
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods each element must have
        n_elements (int): The number of elements

    """

    def __init__(
        self,
        blend_keys: List[str],
        elements: List[Type[Element]],
        name=None,
    ):
        """Hashin-Striktman Averaging Type Frame for two phase systems.

        Elements must implement `density()`, `vp()`, `vs()`, `shear_modulus()`, `bulk_modulus()`, `shear_modulus()`

        Args:
            blend_keys: props keys needed by blending calcs
            elements: length 2 list of elements
            name: Name of the rock frame
        """
        super().__init__(
            blend_keys,
            elements,
            name=name,
        )

    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns bulk modulus

        Args:
            props: A dictionary of properties required.
            **kwargs: passed to elements

        Returns:
            Bulk Modulus
        """
        args = self._process_props_get_method(props, ["bulk_modulus", "shear_modulus"])
        bulk, _ = models.hsw_avg(*args)
        return bulk

    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Returns bulk modulus

        Args:
            props: A dictionary of properties required.
            **kwargs: passed to elements

        Returns:
            Bulk Modulus
        """
        args = self._process_props_get_method(props, ["bulk_modulus", "shear_modulus"])
        _, shear = models.hsw_avg(*args)
        return shear


# class CementedSandFrame(HSFrame):
#     def __init__(self, minerals=None):
#         """Cemented Sand Frame based upon Hashin-Striktman Averaging for two minerals.

#         Args:
#             minerals ([type], optional): [description]. Defaults to None.
#         """
#         super.__init__(minerals=minerals)

#     def _dryframe_csand_moduli(self, porosity, min_kwargs, component="bulk"):
#         """Dryframe moduluii for a cemented sand."""
#         if len(min_kwargs) != 2:
#             raise ValueError(
#                 "Cemented Sand moduli requires exactly two mineral components."
#             )
#         min1, min2 = tuple(min_kwargs.keys())

#         keff, mueff = dryframe_cemented_sand(
#             self.minerals[min1]["bulk"],
#             self.minerals[min1]["shear"],
#             self.minerals[min2]["bulk"],
#             self.minerals[min2]["shear"],
#             self.crit_porosity,
#             porosity,
#             self.ncontacts,
#             alpha=self.csand_scheme,
#         )

#         if component == "bulk":
#             return keff
#         if component == "shear":
#             return mueff

#         raise ValueError("Unknown component: {}".format(component))

#     def dry_frame_bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
#         """No stress sensitivity applied."""
#         min_kwargs = _check_kwargs_vfrac(**min_kwargs)
#         k0 = self._hs_avg_moduli(min_kwargs, component="bulk")
#         kdry = self._dryframe_csand_moduli(porosity, min_kwargs, component="bulk")
#         return kdry

#     def bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
#         """Returns modulus of dry frame at conditions.

#         Args:
#             porosity
#             temp (array-like): Temperature (degC)
#             pres (array-like): Pressure (MPa)

#         Returns:
#             array-like : Modulus for temp and pres (GPa).
#         """
#         kdry = self.dry_frame_bulk_modulus(porosity, **min_kwargs)
#         if self._is_stress_sens and self.calb_pres is None:
#             raise WorkflowError(
#                 "bulk_modulus",
#                 "Calibration pressure for stress dryframe adjustment not set.",
#             )

#         if self._is_stress_sens:
#             kdryf = self.stress_model.stress_dryframe_moduli(
#                 depth, self.calb_pres, pres, kdry, component="bulk"
#             )
#         else:
#             kdryf = kdry

#         return kdryf

#     def dry_frame_shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
#         min_kwargs = _check_kwargs_vfrac(**min_kwargs)
#         return self._dryframe_csand_moduli(porosity, min_kwargs, component="shear")

#     def shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
#         mudry = self.dry_frame_shear_modulus(porosity, **min_kwargs)
#         if self._is_stress_sens and self.calb_pres is None:
#             raise WorkflowError(
#                 "shear_modulus",
#                 "Calibration pressure for stress dryframe adjustment not set.",
#             )

#         if self._is_stress_sens:
#             mudryf = self.stress_model.stress_dryframe_moduli(
#                 depth, self.calb_pres, pres, mudry, component="shear"
#             )
#         else:
#             mudryf = mudry

#         return mudryf
