# pylint:disable=missing-docstring
import sys
from ._version import version as __version__

from ._fluid import (
    FluidModel,
    Fluid,
    Water,
    WaterECL,
    DeadOil,
    OilBW92,
    OilPVT,
    Gas,
    GasECL,
)
from ._fluid_loaders import (
    load_pvtw,
    load_pvto,
    load_pvdg,
)

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
