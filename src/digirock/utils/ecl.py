"""Mappings for key Eclipse Properties
"""
from enum import Enum, unique

# pylint: disable=missing-docstring


class InitIntheadMap(Enum):
    UNITS = 2
    PVTM = 3
    NI = 8
    NJ = 9
    NK = 10
    NACTIV = 11
    GRID_TYPE = 13
    PHASE = 14
    IPROG = 94


@unique
class EclUnitMap(Enum):
    METRIC = 1
    FIELD = 2
    LAB = 3
    PVTM = 4


class EclStandardConditions(Enum):
    TEMP = 15.5556  # degC from 60F
    PRES = 0.101325  # MPa from 1 Atm


class E100MetricConst(Enum):
    PRES_ATMS = 1.013  # barsa
    RHO_AIR = 1.22  # kg/m3
    RHO_WAT = 1000.0  # kg/m3
    GAS_CONST = 0.083143  # m3bars/K/kg-M


class E100GravConst(Enum):
    METRIC = 0.0000981  # m2bars/kg
    FIELD = 0.00694  # ft2psi/lb
    LAB = 0.0009678  # cm2atm/gm
    PVTM = 1


class E300GravConst(Enum):
    METRIC = 0.0000980665  # m2bars/kg
    FIELD = 0.00694444  # ft2psi/lb
    LAB = 0.000967841  # cm2atm/gm
    PVTM = 0.0000967841


class EclUnitScaler(Enum):
    METRIC = dict(
        length=1,
        time=1,
        area=1,
        density=1e-3,
        density_kg=1.0,
        pressure=1e-1,
        ipressure=1 / 1e-1,
        temp_abs=1,
        temp_rel=lambda x: x,
        compress=1e2,
        viscosity=1,
        perm=1,
        volume=1,
        unitless=1,
    )
    FIELD = dict(
        length=0.3048,
        time=1,
        area=10.7639,
        density=0.0160184634,
        density_kg=16.0185,
        pressure=0.00689476,
        ipressure=1 / 0.00689476,
        temp_abs=5 / 9,
        temp_rel=lambda x: x - 32 * 5 / 9,
        compress=1 / 0.00689476,
        viscosity=1,
        perm=1,
        volume=0.158987,
        unitless=1,
    )
    LAB = dict(
        length=1e-2,
        time=1 / 24,
        area=1e-4,
        density=1,
        density_kg=1000.0,
        pressure=0.101325,
        ipressure=1 / 0.101325,
        temp_abs=1,
        temp_rel=lambda x: x,
        compress=1 / 0.101325,
        viscosity=1,
        perm=1,
        volume=1e-6,
        unitless=1,
    )
    PVTM = dict(
        length=1,
        time=1,
        area=1,
        density=1e-3,
        density_kg=1.0,
        pressure=0.101325,
        ipressure=1 / 0.101325,
        temp_abs=1,
        temp_rel=lambda x: x,
        compress=1 / 0.101325,
        viscosity=1,
        perm=1,
        volume=1,
        unitless=1,
    )
