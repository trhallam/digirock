# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Frames
#
# `RockFrame` classes are used in `digirock` to build the dry rock frames. They inherit the `Blend` class.

# %%
from digirock import Blend, Element
from digirock.utils.types import NDArrayOrFloat
from typing import List, Dict, Type
from digirock import Quartz, Feldspar

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
        blend_keys: List[str],
        elements: List[Type[Element]],
        name=None,
    ):
        """
        
        Elements must implement `density()`, `vp()`, `vs()`, `shear_modulus()`, `bulk_modulus()`

        Args:
            blend_keys:
            elements:
            methods:
            name: Name of the rock frame

        """
        methods = ["density", "vp", "vs", "shear_modulus", "bulk_modulus"]
        super().__init__(blend_keys, elements, methods, name=name)


    def density(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns density of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Density (g/cc).
        """
        raise PrototypeError(self.__class__.__name__, "density")
                       
    def vp(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns compression velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Velocity (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "vp")
                       
    def vs(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns shear velocity of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            Velocity (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "vs")

    def bulk_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns bulk modulus of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            bulk modulus (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "bulk_modulus")
                       

    def shear_modulus(self, props: Dict[str, NDArrayOrFloat], **kwargs) -> NDArrayOrFloat:
        """Returns shear modulus of RockFrame

        Args:
            props: A dictionary of properties required.
            kwargs: ignored

        Returns:
            shear modulus (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "shear_modulus")

rf = RockFrame([], [Quartz, Feldspar])




# %%
rf.tree

# %%
Quartz.tree

# %%
