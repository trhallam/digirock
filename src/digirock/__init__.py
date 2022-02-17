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

from ._frames._minerals import Mineral

from ._stress import StressModel, FStressModel, LGStressModel

# from ._frames._poro_adjust import (
#     PoroAdjModel,
#     DefaultPoroAdjModel,
#     NurCriticalPoro,
#     WoodsideCementPoro,
#     LeeConsolodationPoro,
# )
# from ._frames._frames import (
#     RockFrame,
#     VRHFrame,
#     HSFrame,
#     CementedSandFrame,
# )

# from ._rock import Mineral, RockModel, FaciesModel, MultiRockModel

from ._mineral_lib import minerals as _minerals

for mineral, vals in _minerals.items():
    bulk, shear, density = vals
    setattr(sys.modules[__name__], mineral, Mineral(mineral, density, bulk, shear))
