"""Mineral and Rock models to simplify generation of rock properties.

"""
# pylint: disable=invalid-name,bad-continuation

import numpy as np

from ._exceptions import WorkflowError

from ._fluid import FluidModel
from .  _frame import RockFrame
from ._stress import StressModel
from .elastic import acoustic_vel, acoustic_moduli
from .models import gassmann_fluidsub


class Mineral:
    """Base Class for defining Minerals. This are rock constituents.

    Attributes:
        name (str): name of the mineral
        density (float): density of the mineral in g/cc
        bulk_modulus (float): Bulk modulus (compression) of the mineral in GPa
        shear_modulus (float): Shear modulus of the mineral in GPa
    """

    def __init__(self, name=None, density=None, bulk_modulus=None, shear_modulus=None):
        self.name = name
        self.density = density
        self.bulk_modulus = bulk_modulus
        self.shear_modulus = shear_modulus
        self._quick_get = {"bulk": bulk_modulus, "shear": shear_modulus, "rho": density}

    def _check_defined(self, from_func, var):
        if self.__getattribute__(var) is None:
            raise WorkflowError(from_func, f"The {var} attribute is not defined.")

    def __getitem__(self, key):
        try:
            return self._quick_get[key]
        except KeyError:
            raise KeyError(
                f"unknown {key} for mineral expects one of {self._quick_get.keys()}"
            )

    def elastic(self):
        """Pure elastic properties of mineral."""
        vp, vs = acoustic_vel(self.bulk_modulus, self.shear_modulus, self.density)
        return vp, vs, self.density

    def get_summary(self):
        summary = super().get_summary()
        summary.update(
            {
                "name": self.name,
                "k": self.bulk_modulus,
                "mu": self.shear_modulus,
                "rhob": self.density,
            }
        )
        return summary


class RockModel:
    """Build a rock model from minerals, fluids and methods.

    Attributes:
        name (str): name of the rock
        fluid_model (etlpy.pem.FluidModel): The fluid model.
        minerals (dict): Dictionary of mineral components of type (etlpy.pem.Mineral).

    """

    def __init__(
        self,
        name=None,
        fluid_model=None,
        rockframe_model=None,
    ):
        """[summary]

        Args:
            name (str, optional): [description]. Defaults to None.
            fluid_model (etlpy.pem.FluidModel, optional): The rock fluid model. Defaults to None.
            rockframe_model (etlpy.pem.RockFrame, optional): The rock frame model. Defaults to None.
        """
        self.name = name

        if fluid_model is not None:
            self.set_fluid_model(fluid_model)
        else:
            self.fluid_model = None

        if rockframe_model is not None:
            self.set_rockframe_model(rockframe_model)
        else:
            self.rockframe_model = None

    def set_fluid_model(self, fluid_model):
        """Set the fluid model. Must be of type etlpy.pem.FluidModel"""
        if isinstance(fluid_model, FluidModel):
            self.fluid_model = fluid_model
        else:
            raise ValueError(
                f"fluid_model should be of type {FluidModel} got {type(fluid_model)}"
            )

    def set_rockframe_model(self, rockframe_model):
        """Set the RockFrame model. Must be of type etlpy.pem.RockFrame"""
        if isinstance(rockframe_model, RockFrame):
            self.rockframe_model = rockframe_model
        else:
            raise ValueError(
                f"fluid_model should be of type {RockFrame} got {type(rockframe_model)}"
            )

    def _split_kwargs(self, **kwargs):
        if not kwargs:
            raise ValueError(
                f"Specify at least one fluid satursation"
                "and one mineral volume to mix."
            )

        fluid_names = tuple(self.fluid_model.components.keys())
        mineral_names = tuple(self.rockframe_model.minerals.keys())

        fluid_keys = list()
        mineral_keys = list()
        unknown_keys = list()

        fluids_known_extras = ("bo", "rs")

        for key in kwargs:
            if key in fluid_names + fluids_known_extras:
                fluid_keys.append(key)
            elif key in mineral_names:
                mineral_keys.append(key)
            else:
                unknown_keys += [key]

        if unknown_keys:
            raise ValueError(f"Unknown mineral and fluid keywords found {unknown_keys}")

        min_kwargs = {key: kwargs[key] for key in mineral_keys}
        fluid_kwargs = {key: kwargs[key] for key in fluid_keys}
        return min_kwargs, fluid_kwargs

    def density(self, porosity, temp, pres, depth, **kwargs):
        """Return the density of the rock

        Args:
            porosity (array-like): [description]
            kwargs (array-like):
        """
        # must be at least one fluid and one mineral component
        min_kwargs, fluid_kwargs = self._split_kwargs(**kwargs)
        fluid_density = self.fluid_model.density(temp, pres, **fluid_kwargs)
        min_dens = self.rockframe_model.density(
            porosity, temp, pres, depth, **min_kwargs
        )
        return porosity * fluid_density + min_dens

    def bulk_modulus(self, porosity, temp, pres, depth, **kwargs):
        """Return the bulk modulus of the rock

        Special behaviour:
            If no fluid components are specified the dry frame modulus will be returned.

        Args:
            porosity (array-like): Porosity values
            temp (array-like): Temperature
            pres (array-like): Pressure-values
            use_stress_sens (bool): Use the stress sensitivity equations to adjust moduli for
                pressure. Defaults to False.
            depth (array-like): The depth for stress calculations. Defaults to None.
            kwargs (array-like): Volume and saturation fractions for minerals and fluids.

        Returns:
            array-like: Bulk modulus for input values.
        """
        min_kwargs, fluid_kwargs = self._split_kwargs(**kwargs)

        if fluid_kwargs:
            kfl = self.fluid_model.modulus(temp, pres, **fluid_kwargs)
        else:
            kfl = None  # will return the dry frame modulus

        k0 = self.rockframe_model.dry_frame_bulk_modulus(
            0.0, temp, pres, depth, **min_kwargs
        )
        kdryf = self.rockframe_model.bulk_modulus(
            porosity, temp, pres, depth, **min_kwargs
        )

        # fluid substitution
        if kfl is None:
            return kdryf
        else:
            return gassmann_fluidsub(kdryf, kfl, k0, porosity)

    def shear_modulus(self, porosity, temp, pres, depth, **kwargs):
        """Return the shear modulus of the rock

        Special behaviour:
            Fluid components are ignored for the shear modulus.

        Args:
            porosity (array-like): Porosity values
            pres (array-like): Pressure-values
            use_stress_sens (bool): Use the stress sensitivity equations to adjust moduli for
                pressure. Defaults to False.
            kwargs (array-like): Volume fractions for minerals.

        Returns:
            array-like: Shear modulus for input values.
        """
        min_kwargs, _ = self._split_kwargs(**kwargs)
        mudryf = self.rockframe_model.shear_modulus(
            porosity, None, pres, depth, **min_kwargs
        )

        return mudryf

    def elastic(
        self,
        porosity,
        temp,
        pres,
        depth,
        output=("velp", "vels", "density"),
        **kwargs,
    ):
        """[summary]

        Args:
            porosity ([type]): [description]
            temp ([type]): [description]
            pres ([type]): [description]
            use_stress_sens (bool, optional): [description]. Defaults to False.
            output (list, optional): [description]. Defaults to ['velp', 'vels', 'density'].

        Raises:
            ValueError: [description]

        Returns:
            list: [description]
        """
        valid_output = ["velp", "vels", "density", "k", "mu", "pimp", "simp"]
        for o in output:
            if o not in valid_output:
                raise ValueError(f"output kw limited to {valid_output} got {o}")
        out = dict()
        out["density"] = self.density(porosity, temp, pres, depth, **kwargs)
        out["k"] = self.bulk_modulus(porosity, temp, pres, depth, **kwargs)
        out["mu"] = self.shear_modulus(porosity, temp, pres, depth, **kwargs)
        out["velp"], out["vels"] = acoustic_vel(out["k"], out["mu"], out["density"])
        if "pimp" in output:
            out["pimp"] = out["velp"] * out["density"]
        if "simp" in output:
            out["simp"] = out["vels"] * out["density"]
        return [out[o] for o in output]

    def get_summary(self):
        summary = {
            "minerals": {
                minr: self.rockframe_model.minerals[minr].get_summary()
                for minr in self.rockframe_model.minerals
            }
        }
        summary.update({"fluids": self.fluid_model.get_summary()})
        summary.update(
            {
                "crit_porosity": self.crit_porosity,
                "porosity_model": self.porosity_model,
                "physics_model": self.physics_model,
                "cons_alpha": self.cons_alpha,
                "cons_gamma": self.cons_gamma,
                "calb_pres": self.calb_pres,
                "csand_scheme": self.csand_scheme,
                "ncontacts": self.ncontacts,
            }
        )
        return summary

    def get_fluid_keys(self):
        """Return a list of all fluid component keys."""
        return list(self.fluid_model.components.keys())

    def get_mineral_keys(self):
        """Return a list of all mineral component keys."""
        return list(self.minerals.keys())


class FaciesModel:
    """A Facies Model that allows you to define simple multiple rock models.

    Args:
        BaseDataModel ([type]): [description]
    """

    def __init__(self, facies):
        """
        Args:
            facies (dict): Dict of etlpy.pem.RockModel
        """
        self.facies = facies

    def elastic(
        self,
        porosity,
        temp,
        pres,
        facies,
        use_stress_sens=False,
        depth=None,
        output=("velp", "vels", "density"),
        **fluid_kwargs,
    ):
        """[summary]

        Args:
            porosity ([type]): [description]
            temp ([type]): [description]
            pres ([type]): [description]
            use_stress_sens (bool, optional): [description]. Defaults to False.
            output (list, optional): [description]. Defaults to ['velp', 'vels', 'density'].

        Raises:
            ValueError: [description]

        Returns:
            list: [description]
        """
        valid_output = ["velp", "vels", "density", "k", "mu", "pimp", "simp"]
        for o in output:
            if o not in valid_output:
                raise ValueError(f"output kw limited to {valid_output} got {o}")
        out = dict()

        nfacies = np.nanmax(facies)
        if nfacies > len(self.facies):
            raise ValueError(
                f"There were more facies in the input {nfacies} "
                f"than in the model {len(self.facies)}."
            )

        size_of_output = np.size(facies)
        dtypes = [(o, float) for o in output]
        out = np.empty(size_of_output, dtype=dtypes)
        out[:] = tuple([np.nan] * len(output))

        for i, (name, fac) in enumerate(self.facies.items()):
            loc_kwargs = fluid_kwargs.copy()
            loc_kwargs[name] = 1
            temp_out = fac.elastic(
                porosity,
                temp,
                pres,
                use_stress_sens=use_stress_sens,
                depth=depth,
                output=output,
                **loc_kwargs,
            )
            for v, op in zip(temp_out, output):
                out[op][facies == i] = v[facies == i]

        return [out[op] for op in output]


class MultiRockModel:
    """A Multi-Rock Model that allows you to define complex reservoir models using multiple rocks
        with a facies flag. Completely different rocks can be defined for each facies e.g. Fluid,
        Stress-Regeime, Mineral Components and or Mineral Models.

    Args:
        BaseDataModel ([type]): [description]
    """

    def set_facies(self, facies):
        """Set the facies of the MultiRockModel"""
        self.facies = facies

    def fluid_density(self, temp, pres, facies, **kwargs):
        """Return fluid density for MultiRockModel"""

        out = np.zeros_like(facies)

        for i, (_, fac) in enumerate(self.facies.items()):
            # get active minerals and fluids for each facies
            active_fluids = fac.fluid_model.components.keys()
            fl_kwargs = {
                fl_key: val for fl_key, val in kwargs.items() if fl_key in active_fluids
            }
            fluid_density = fac.fluid_model.density(
                temp, pres, **{key: kwargs[key] for key in fl_kwargs}
            )
            out = np.where(facies == i, fluid_density, out)
        return out

    def elastic(
        self,
        porosity,
        temp,
        pres,
        facies,
        use_stress_sens=False,
        depth=None,
        output=("velp", "vels", "density"),
        **kwargs,
    ):
        """[summary]

        Args:
            porosity ([type]): [description]
            temp ([type]): [description]
            pres ([type]): [description]
            use_stress_sens (bool, optional): [description]. Defaults to False.
            output (list, optional): [description]. Defaults to ['velp', 'vels', 'density'].
            kwargs (array-like): Mineral and Fluid key words to be used for modelling.
                The appropriate, minerals and fluids will be passed to the sub-models.

        Raises:
            ValueError: [description]

        Returns:
            list: [description]
        """
        valid_output = ["velp", "vels", "density", "k", "mu", "pimp", "simp"]
        for o in output:
            if o not in valid_output:
                raise ValueError(f"output kw limited to {valid_output} got {o}")
        out = dict()

        nfacies = np.nanmax(facies)
        if nfacies > len(self.facies):
            raise ValueError(
                f"There were more facies in the input {nfacies} "
                f"than in the model {len(self.facies)}."
            )

        size_of_output = np.size(facies)
        dtypes = [(o, float) for o in output]
        out = np.empty(size_of_output, dtype=dtypes)
        out[:] = tuple([np.nan] * len(output))

        for i, (_, fac) in enumerate(self.facies.items()):
            # get active minerals and fluids for each facies
            active_minerals = fac.minerals.keys()
            active_fluids = fac.fluid_model.components.keys()
            min_kwargs = {
                min_key: val
                for min_key, val in kwargs.items()
                if min_key in active_minerals
            }
            fl_kwargs = {
                fl_key: val for fl_key, val in kwargs.items() if fl_key in active_fluids
            }

            # create output elastic properties and merge on facies
            temp_out = fac.elastic(
                porosity,
                temp,
                pres,
                use_stress_sens=use_stress_sens,
                depth=depth,
                output=output,
                **min_kwargs,
                **fl_kwargs,
            )
            for v, op in zip(temp_out, output):
                out[op][facies == i] = v[facies == i]

        return [out[op] for op in output]

    def get_fluid_keys(self):
        """Returns a list of all fluid keys in the model.

        Returns:
            list: Full list of fluid keys including duplicates.
        """
        components = list()
        for key in self.facies:
            components += self.facies[key].get_fluid_keys()
        return components

    def get_mineral_keys(self):
        """Returns a list of all mineral keys in the model.

        Returns:
            list: Full list of mineral keys including duplicates.
        """
        components = list()
        for key in self.facies:
            components += self.facies[key].get_mineral_keys()
        components += ["facies"]
        return components


# def elastic_frm(self, porosity, temp1, temp2, pres1, pres2, sat1, sat2,
#                 velp, vels, rhob, use_stress_sens=False, depth=None, **kwargs):
#     """Elastic Fluid Replacement Modelling

#     Args:
#         porosity ([type]): [description]
#         temp1 ([type]): [description]
#         temp2 ([type]): [description]
#         pres1 ([type]): [description]
#         pres2 ([type]): [description]
#         sat1 ([type]): [description]
#         sat2 ([type]): [description]
#         velp ([type]): [description]
#         vels ([type]): [description]
#         rhob ([type]): [description]
#         use_stress_sens (bool, optional): [description]. Defaults to False.
#         depth ([type], optional): [description]. Defaults to None.

#     Returns:
#         tuple: Tuple of vp, vs, and rhob with FRM from sat1 to sat2 applied.
#     """

#     min_kwargs, _ = self._split_kwargs(**kwargs)
#     min_kwargs = self._check_kwargs(**{key:kwargs[key] for key in min_kwargs})

#     ke, mue = mod.acoustic_moduli(velp, vels, rhob)
#     k0 = self._vrh_avg_moduli(min_kwargs, component='bulk')
#     kdry = mod.dryframe_acoustic(ke, self.fluid_model.modulus(temp1, pres1, **sat1), k0, porosity)
#     ke_sub = mod.gassmann_fluidsub(kdry, self.fluid_model.modulus(temp2, pres2, **sat2), k0,
#                                    porosity)
#     rho_dry = mod.dryframe_density(rhob, self.fluid_model.density(temp1, pres1, **sat1),
#                                    porosity)
#     rhob_sub = mod.saturated_density(rho_dry, self.fluid_model.density(temp2, pres2, **sat2),
#                                      porosity)
#     vp_sub, vs_sub = mod.acoustic_vel(ke_sub, mue, rhob_sub)

#     rhob_sub = np.where(np.isnan(rhob_sub), rhob, rhob_sub)
#     vp_sub = np.where(np.isnan(vp_sub), velp, vp_sub)
#     vs_sub = np.where(np.isnan(vs_sub), vels, vs_sub)

#     return vp_sub, vs_sub, rhob_sub
