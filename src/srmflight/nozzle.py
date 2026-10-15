"""Nozzle performance and throat erosion.

Thrust coefficient for a choked, isentropic nozzle (Sutton & Biblarz, ch. 3):

    Cf = sqrt( 2 g^2/(g-1) * (2/(g+1))**((g+1)/(g-1)) * (1 - (pe/p0)**((g-1)/g)) )
         + eps * (pe - pa) / p0

Throat erosion uses a reduced-order power law in chamber pressure. Acharya & Kuo
(2007, JPP 23(6)) found graphite throat recession grows almost linearly with
pressure from 7 to 55 MPa, so the default exponent is m = 1 and the coefficient
``k`` is a calibration / uncertainty parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def area_ratio(mach: float, gamma: float) -> float:
    """Isentropic A/A* at Mach number ``mach``."""
    g = gamma
    return (1.0 / mach) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * mach**2)) ** (
        (g + 1.0) / (2.0 * (g - 1.0))
    )


def exit_mach(expansion_ratio, gamma: float):
    """Supersonic exit Mach number for A_e/A_t (vectorised bisection; scalar or array)."""
    eps = np.asarray(expansion_ratio, dtype=float)
    lo, hi = np.ones_like(eps), np.full_like(eps, 50.0)
    for _ in range(60):  # A/A* is monotonic for M > 1; 60 halvings -> ~1e-16 relative
        mid = 0.5 * (lo + hi)
        too_small = area_ratio(mid, gamma) < eps
        lo, hi = np.where(too_small, mid, lo), np.where(too_small, hi, mid)
    me = np.where(eps <= 1.0, 1.0, 0.5 * (lo + hi))
    return float(me) if me.ndim == 0 else me


def thrust_coefficient(gamma: float, expansion_ratio, p0, pa: float):
    """Ideal thrust coefficient Cf for chamber pressure ``p0`` and ambient ``pa`` [Pa]."""
    g = gamma
    p0 = np.asarray(p0, dtype=float)
    me = exit_mach(expansion_ratio, g)
    pe_p0 = (1.0 + 0.5 * (g - 1.0) * me**2) ** (-g / (g - 1.0))
    momentum = np.sqrt(
        2.0 * g**2 / (g - 1.0)
        * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
        * (1.0 - pe_p0 ** ((g - 1.0) / g))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = momentum + np.asarray(expansion_ratio) * (pe_p0 - pa / p0)
    cf = np.where(p0 > 0.0, cf, 0.0)
    return float(cf) if cf.ndim == 0 else cf


@dataclass(frozen=True)
class ThroatErosion:
    """Throat-radius recession  dr_t/dt = k * (p / p_ref)**m  [m/s]."""

    k: float = 0.0
    m: float = 1.0
    p_ref: float = 1.0e6

    def rate(self, p: float) -> float:
        if self.k == 0.0 or p <= 0.0:
            return 0.0
        return self.k * (p / self.p_ref) ** self.m


@dataclass(frozen=True)
class Nozzle:
    """Conical/bell nozzle described by throat and exit diameters.

    ``efficiency`` multiplies the ideal Cf (divergence, friction, two-phase losses).
    """

    throat_diameter: float
    exit_diameter: float
    efficiency: float = 1.0
    erosion: ThroatErosion = ThroatErosion()

    @property
    def throat_area(self) -> float:
        return 0.25 * np.pi * self.throat_diameter**2

    @property
    def exit_area(self) -> float:
        return 0.25 * np.pi * self.exit_diameter**2

    def thrust(self, p0, throat_radius, gamma: float, pa: float):
        """Thrust [N] for chamber pressure ``p0`` and current (eroded) throat radius.

        Accepts arrays; thrust is zero once the chamber is at or below ambient.
        """
        p0 = np.asarray(p0, dtype=float)
        at = np.pi * np.asarray(throat_radius, dtype=float) ** 2
        eps = self.exit_area / at
        F = self.efficiency * thrust_coefficient(gamma, eps, p0, pa) * p0 * at
        F = np.where(p0 > pa, np.maximum(F, 0.0), 0.0)
        return float(F) if F.ndim == 0 else F
