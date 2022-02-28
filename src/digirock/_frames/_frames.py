"""Classes for various RockFrames, new RockFrames should inherit `RockFrame` as a base class.
"""
from typing import Sequence, Dict, Type, List, Union, Tuple
from .._base import Blend, Element
from ..utils._decorators import check_props
from ..utils._utils import _process_vfrac
from ..typing import NDArrayOrFloat, NDArrayOrInt
from .. import models
from ..elastic import acoustic_vels, acoustic_velp
from ._minerals import Mineral


class RockFrame(Blend):
    """Base Class for defining rock frame models all new rock frames should be based upon this class.

    Attributes:
        name (str): name of the model
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
        name (str): name of the model
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
            name: Name of the model
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
        name (str): name of the model
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
            name: Name of the model
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


class CementedSand(RockFrame):
    """Cemented Sand dry rock for sand and cement system

    Attributes:
        name (str): name of the model
        blend_keys (list): Keys to use for blending
        elements (list): A list of elements
        methods (list): A list of methods each element must have
        n_elements (int): The number of elements

    """

    def __init__(
        self,
        sand_vfrac_key: str,
        sand: Type[Mineral],
        cement_vfrac_key: str,
        cement: Type[Mineral],
        ncontacts: int = 9,
        alpha: Union[str, float] = "scheme1",
        name=None,
    ):
        """Cemented Sand dry rock Type Frame for two phase systems (Sand, Cement).

        Elements must implement `density()`, `vp()`, `vs()`, `shear_modulus()`, `bulk_modulus()`, `shear_modulus()`

        Args:
            sand_vfrac_key: props key for sand
            sand: sand element
            cement_vfrac_key: props key for cement
            cement: cement element
            ncontacts: number of sand grain contacts
            alpha: the cemented sand scheme to use
            name: name of the model
        """
        super().__init__([sand_vfrac_key, cement_vfrac_key], [sand, cement], name=name)
        self._ncontacts = ncontacts
        self._alpha = alpha

    @property
    def ncontacts(self):
        return self._ncontacts

    @property
    def alpha(self):
        return self._alpha

    def _element_moduli(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> Tuple[NDArrayOrFloat, NDArrayOrFloat, NDArrayOrFloat, NDArrayOrFloat]:
        """Returns the moduli for the two elements."""
        k_sand = self.elements[0].bulk_modulus(props, **kwargs)
        mu_sand = self.elements[0].shear_modulus(props, **kwargs)
        k_cement = self.elements[1].bulk_modulus(props, **kwargs)
        mu_cement = self.elements[1].shear_modulus(props, **kwargs)
        return k_sand, mu_sand, k_cement, mu_cement

    def _phi0(self, props: Dict[str, NDArrayOrFloat]) -> NDArrayOrFloat:
        """Calculate porosity for sand grains only

        phi0 = phi + vfrac_cement
        """
        return props["poro"] + props[self.blend_keys[1]]

    def _get_ncontacts(self, props: Dict[str, NDArrayOrFloat]) -> NDArrayOrInt:
        if "ncontacts" in props:
            ncontacts = props["ncontacts"]
        else:
            ncontacts = self.ncontacts
        return ncontacts

    @check_props("poro", broadcastable=("poro", "ncontacts"))
    def bulk_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Bulk modulus for Cemented Sand"""
        phi0 = self._phi0(props)
        mods = self._element_moduli(props, **kwargs)
        ncontacts = self._get_ncontacts(props)

        k, _ = models.dryframe_cemented_sand(
            *mods,
            phi0,
            props["poro"],
            ncontacts=ncontacts,
            alpha=self.alpha,
        )
        return k

    @check_props("poro", broadcastable=("poro", "ncontacts"))
    def shear_modulus(
        self, props: Dict[str, NDArrayOrFloat], **kwargs
    ) -> NDArrayOrFloat:
        """Shear modulus for Cemented Sand"""
        phi0 = self._phi0(props)
        mods = self._element_moduli(props, **kwargs)
        ncontacts = self._get_ncontacts(props)

        _, mu = models.dryframe_cemented_sand(
            *mods,
            phi0,
            props["poro"],
            ncontacts=ncontacts,
            alpha=self.alpha,
        )
        return mu

    @check_props("poro")
    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Frame density accounting for porosity"""
        # add porosity to front of args (has zero density)
        args = (0.0, props["poro"]) + self._process_props_get_method(props, "density")
        return models.mixed_density(*args)
