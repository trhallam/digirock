# pylint:disable=missing-docstring
import sys

from ._fluid import FluidModel, Fluid, Water, WaterECL, DeadOil, Oil, Gas, GasPVDG

from ._frame import (
    PoroAdjModel,
    DefaultPoroAdjModel,
    NurCriticalPoro,
    WoodsideCementPoro,
    LeeConsolodationPoro,
    RockFrame,
    VRHFrame,
    HSFrame,
    CementedSandFrame,
)

from ._rock import Mineral, RockModel, FaciesModel, MultiRockModel
from ._stress import StressModel
from ._mineral_lib import minerals as _minerals

for mineral, vals in _minerals.items():
    bulk, shear, density = vals
    setattr(sys.modules[__name__], mineral, Mineral(mineral, density, bulk, shear))
