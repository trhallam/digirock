# pylint:disable=missing-docstring
import sys
from ._version import version as __version__

from ._base import Element, Switch, Blend, Transform

from ._fluids import *
from ._frames import *

from ._stress import StressModel, FStressModel, LGStressModel

from ._rock import GassmannRock

from ._mineral_lib import minerals as _minerals

for mineral, vals in _minerals.items():
    bulk, shear, density = vals
    setattr(sys.modules[__name__], mineral, Mineral(density, bulk, shear, name=mineral))
