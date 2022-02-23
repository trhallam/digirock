"""Classes for various RockFrames, new RockFrames should inherit `RockFrame` as a base class.
"""


class RockFrame:
    """Base Class for defining rock frame models all new rock frames should be based upon this class.

    Attributes:
        name (str): name of the fluid
    """

    def __init__(
        self,
        name=None,
        minerals=None,
        porosity_adjust=None,
        stress_model=None,
    ):
        """[summary]

        Args:
            name (str, optional): [description]. Defaults to None.
            minerals (list/dict, optional): Mineral components of the FrameModel. Defaults to None.

        """
        save_header = "etlpy RockFrame class instance save file"
        super().__init__(save_header)
        self.name = name

        self.minerals = None
        if minerals is None:
            pass
        elif isinstance(minerals, list):
            for mineral in minerals:
                self.add_mineral(mineral.name, mineral)
        elif isinstance(minerals, dict):
            self.minerals = minerals
        else:
            raise ValueError(
                "minerals should be of type "
                f"{(type(list), type(dict))} got {type(minerals)}"
            )

        if porosity_adjust is None:
            self.porosity_adjust = DefaultPoroAdjModel("defaulted_poroadj")
        else:
            self.porosity_adjust = porosity_adjust

        if stress_model is None:
            self.stress_model = None
            self._is_stress_sens = False
        else:
            self.set_stress_model(stress_model)
            self._is_stress_sens = True

    def set_porosity_adjust_model(self, porosity_adjust):
        """Set the porosity adjustment model"""
        if not isinstance(porosity_adjust, PoroAdjModel):
            raise ValueError(
                f"porosity adjustment model should be of type {PoroAdjModel}"
            )
        self.porosity_adjust = porosity_adjust

    def set_stress_model(self, stress_model, calibration_pressure):
        if isinstance(stress_model, StressModel):
            self.stress_model = stress_model
            if self.stress_model._has_stress_sens:
                self._is_stress_sens = True
        elif stress_model is not None:
            raise ValueError(
                f"stress model must be {StressModel} got {type(stress_model)}"
            )
        self.calb_pres = calibration_pressure

    def _check_defined(self, from_func, var):
        if self.__getattribute__(var) is None:
            raise WorkflowError(from_func, f"The {var} attribute is not defined.")

    def add_mineral(self, name, mineral):
        """Add a mineral component to the rock model.

        Args:
            name (str): Name of the mineral
            mineral (etlp.pem.Mineral): Mineral component class.
        """
        if self.minerals is None:
            self.minerals = dict()
        self.minerals[name] = mineral

    def reuss_bound(self, **kwargs):
        """Returns the reuss bounds for mineral components."""
        kwargs = _check_kwargs_vfrac(**kwargs)
        mods = [self.minerals[key].bulk_modulus for key in kwargs]
        fracs = [val for val in kwargs.values()]
        fracs = np.array(fracs).T
        return _mod.reuss_lower_bound(mods, fracs)

    def density(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns density of mineral frame at porosity, temp and pres.

        By default this is bulk density with vacuum in the porespace.

        Temp and pres are included for sub-classing but not used by this implementation.

        Args:
            porosity (array-like): Porosity vfrac (0-1)
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Density for porosity, temp and pres (g/cc).
        """
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        n_min = len(min_kwargs)
        min_frac = np.zeros((np.array(porosity).size, n_min))
        i = 0
        for minkw, vals in min_kwargs.items():
            min_frac[:, i] = self.minerals[minkw]["rho"] * vals
            i = i + 1
        min_dens = min_frac.sum(axis=1)
        return (1 - porosity) * min_dens

    def velocity(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns density of fluid at temp and pres.

        Args:
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Velocity for temp and pres (m/s).
        """
        raise PrototypeError(self.__class__.__name__, "velocity")

    def dry_frame_bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """"""
        raise PrototypeError(self.__class__.__name__, "modulus")

    def bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns modulus of fluid at temp and pres.

        Args:
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Modulus for temp and pres (GPa).
        """
        raise PrototypeError(self.__class__.__name__, "modulus")

    def dry_frame_shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        raise PrototypeError(self.__class__.__name__, "modulus")

    def shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        raise PrototypeError(self.__class__.__name__, "modulus")

    def get_summary(self):
        """Return a dictionary containing a summary of the fluid.

        Returns:
            dict: Summary of properties.
        """
        summary = super().get_summary()
        summary.update({"name": self.name})
        return summary


class VRHFrame(RockFrame):
    def __init__(self, minerals=None):
        """Voight-Reuss-Hill Averaging Type Frame for two minerals.

        Args:
            minerals ([type], optional): [description]. Defaults to None.
        """
        super().__init__(minerals=minerals)

    def _vrh_avg_moduli(self, min_kwargs, component="bulk"):
        """VRH avg moduli helper

        Args:
            min_kwargs (dict): Two mineral compontents to mix.
        """
        if len(min_kwargs) != 2:
            raise ValueError("VRH averaging requires exactly two mineral components.")

        min1, min2 = tuple(min_kwargs.keys())
        m0 = _mod.vrh_avg(
            self.minerals[min1][component],
            self.minerals[min2][component],
            min_kwargs[min1],
        )
        return m0

    def dry_frame_bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """No stress sensitivity applied."""
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        k0 = self._vrh_avg_moduli(min_kwargs, component="bulk")
        phi_fact = self.porosity_adjust.transform(
            porosity, component="bulk", **min_kwargs
        )
        return k0 * phi_fact

    def bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns modulus of dry frame at conditions.

        Args:
            porosity
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Modulus for temp and pres (GPa).
        """
        kdry = self.dry_frame_bulk_modulus(porosity, temp, pres, depth, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "bulk_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            kdryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, kdry, component="bulk"
            )
        else:
            kdryf = kdry

        return kdryf

    def dry_frame_shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        mu0 = self._vrh_avg_moduli(min_kwargs, component="shear")
        phi_fact = self.porosity_adjust.transform(
            porosity, component="shear", **min_kwargs
        )
        return mu0 * phi_fact

    def shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        mudry = self.dry_frame_shear_modulus(porosity, temp, pres, depth, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "shear_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            mudryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, mudry, component="shear"
            )
        else:
            mudryf = mudry

        return mudryf


class HSFrame(RockFrame):
    def __init__(self, minerals=None):
        """Hashin-Striktman Averaging Type Frame for two minerals.

        Args:
            minerals ([type], optional): [description]. Defaults to None.
        """
        super().__init__(minerals=minerals)

    def _hs_avg_moduli(self, min_kwargs, component="bulk"):
        """Hashin-Striktman bulk modulus

        Args:
            min_kwargs (dict): Two mineral components to mix.
        """
        if len(min_kwargs) != 2:
            raise ValueError("HS moduli requires exactly two mineral components.")
        min1, min2 = tuple(min_kwargs.keys())

        if component == "bulk":
            hsb1 = _mod.hs_kbounds(
                self.minerals[min1]["bulk"],
                self.minerals[min2]["bulk"],
                self.minerals[min1]["shear"],
                min_kwargs[min1],
            )
            hsb2 = _mod.hs_kbounds(
                self.minerals[min2]["bulk"],
                self.minerals[min1]["bulk"],
                self.minerals[min2]["shear"],
                min_kwargs[min1],
            )
        elif component == "shear":
            hsb1 = _mod.hs_mubounds(
                self.minerals[min1]["bulk"],
                self.minerals[min1]["shear"],
                self.minerals[min2]["shear"],
                min_kwargs[min1],
            )
            hsb2 = _mod.hs_mubounds(
                self.minerals[min2]["bulk"],
                self.minerals[min1]["shear"],
                self.minerals[min2]["shear"],
                min_kwargs[min1],
            )
        else:
            raise ValueError(f"unknown component keyword {component}")
        return 0.5 * (hsb1 + hsb2)

    def dry_frame_bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """No stress sensitivity applied."""
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        k0 = self._hs_avg_moduli(min_kwargs, component="bulk")
        phi_fact = self.porosity_adjust.transform(
            porosity, component="bulk", **min_kwargs
        )
        return k0 * phi_fact

    def bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns modulus of dry frame at conditions.

        Args:
            porosity
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Modulus for temp and pres (GPa).
        """
        kdry = self.dry_frame_bulk_modulus(porosity, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "bulk_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            kdryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, kdry, component="bulk"
            )
        else:
            kdryf = kdry

        return kdryf

    def dry_frame_shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        mu0 = self._hs_avg_moduli(min_kwargs, component="shear")
        phi_fact = self.porosity_adjust.transform(
            porosity, component="shear", **min_kwargs
        )
        return mu0 * phi_fact

    def shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        mudry = self.dry_frame_shear_modulus(porosity, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "shear_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            mudryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, mudry, component="shear"
            )
        else:
            mudryf = mudry

        return mudryf


class CementedSandFrame(HSFrame):
    def __init__(self, minerals=None):
        """Cemented Sand Frame based upon Hashin-Striktman Averaging for two minerals.

        Args:
            minerals ([type], optional): [description]. Defaults to None.
        """
        super.__init__(minerals=minerals)

    def _dryframe_csand_moduli(self, porosity, min_kwargs, component="bulk"):
        """Dryframe moduluii for a cemented sand."""
        if len(min_kwargs) != 2:
            raise ValueError(
                "Cemented Sand moduli requires exactly two mineral components."
            )
        min1, min2 = tuple(min_kwargs.keys())

        keff, mueff = dryframe_cemented_sand(
            self.minerals[min1]["bulk"],
            self.minerals[min1]["shear"],
            self.minerals[min2]["bulk"],
            self.minerals[min2]["shear"],
            self.crit_porosity,
            porosity,
            self.ncontacts,
            alpha=self.csand_scheme,
        )

        if component == "bulk":
            return keff
        if component == "shear":
            return mueff

        raise ValueError("Unknown component: {}".format(component))

    def dry_frame_bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """No stress sensitivity applied."""
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        k0 = self._hs_avg_moduli(min_kwargs, component="bulk")
        kdry = self._dryframe_csand_moduli(porosity, min_kwargs, component="bulk")
        return kdry

    def bulk_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        """Returns modulus of dry frame at conditions.

        Args:
            porosity
            temp (array-like): Temperature (degC)
            pres (array-like): Pressure (MPa)

        Returns:
            array-like : Modulus for temp and pres (GPa).
        """
        kdry = self.dry_frame_bulk_modulus(porosity, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "bulk_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            kdryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, kdry, component="bulk"
            )
        else:
            kdryf = kdry

        return kdryf

    def dry_frame_shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        min_kwargs = _check_kwargs_vfrac(**min_kwargs)
        return self._dryframe_csand_moduli(porosity, min_kwargs, component="shear")

    def shear_modulus(self, porosity, temp, pres, depth, **min_kwargs):
        mudry = self.dry_frame_shear_modulus(porosity, **min_kwargs)
        if self._is_stress_sens and self.calb_pres is None:
            raise WorkflowError(
                "shear_modulus",
                "Calibration pressure for stress dryframe adjustment not set.",
            )

        if self._is_stress_sens:
            mudryf = self.stress_model.stress_dryframe_moduli(
                depth, self.calb_pres, pres, mudry, component="shear"
            )
        else:
            mudryf = mudry

        return mudryf
