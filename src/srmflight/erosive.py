"""Erosive-burning models: burn-rate augmentation by the core cross-flow.

All models expose ``total_rate(r0, G, d_char, x_head, propellant)`` returning the
augmented burn rate r [m/s] given the non-erosive rate r0 [m/s], local port mass
flux G [kg/m^2/s], characteristic port length d_char [m] (hydraulic diameter) and
distance from the head end x_head [m].

Lenoir & Robillard (1957), additive heat-transfer model:

    r = r0 + alpha * G**0.8 / L**0.2 * exp(-beta * rho_p * r / G)

r appears on both sides, so it is solved by bracketed root finding. The original
paper uses L = distance from the head end; Lawrence et al. (1968) replaced it with
the hydraulic diameter, which Willcox et al. (2007) report behaves better for
large motors. Both are available through ``length_scale``.

Mukunda & Paul (1997), universal correlation:

    eta = r / r0 = 1 + 0.023 * (g**0.8 - g_th**0.8) * H(g - g_th)
    g = g0 * (Re0 / 1000)**-0.125,   g0 = G / (rho_p r0),   Re0 = rho_p r0 d / mu
    g_th = 35
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class NoErosion:
    """Erosive burning switched off."""

    def total_rate(self, r0, G, d_char, x_head, propellant):  # noqa: D401
        return r0


@dataclass(frozen=True)
class LenoirRobillard:
    """Lenoir-Robillard erosive burning in SI units.

    Parameters
    ----------
    alpha : erosive coefficient [SI: m^2.8 kg^-0.8 s^-0.2]
    beta : blowing-correction exponent [-]; ~53 and reported as propellant independent
    length_scale : "hydraulic_diameter" (Lawrence modification) or "head_distance" (original)
    """

    alpha: float
    beta: float = 53.0
    length_scale: Literal["hydraulic_diameter", "head_distance"] = "hydraulic_diameter"

    @classmethod
    def from_properties(
        cls,
        cp_gas: float,
        mu_gas: float,
        pr_gas: float,
        rho_p: float,
        cp_solid: float,
        T_flame: float,
        T_surface: float,
        T_grain: float,
        beta: float = 53.0,
        length_scale: Literal["hydraulic_diameter", "head_distance"] = "hydraulic_diameter",
    ) -> "LenoirRobillard":
        """alpha from a Chilton-Colburn heat balance (Lenoir & Robillard 1957, Eq. 16).

        alpha = 0.0288 cp mu^0.2 Pr^(-2/3) / (rho_p c_s) * (T_f - T_s) / (T_s - T_i)

        Because alpha scales with 1/(T_s - T_i), a hotter grain raises both the
        base and the erosive burn rate - the coupling used in the UQ study.
        """
        alpha = (
            0.0288 * cp_gas * mu_gas**0.2 * pr_gas ** (-2.0 / 3.0) / (rho_p * cp_solid)
            * (T_flame - T_surface) / (T_surface - T_grain)
        )
        return cls(alpha=alpha, beta=beta, length_scale=length_scale)

    def erosive_increment(self, r, G, L, rho_p):
        if G <= 0.0 or L <= 0.0:
            return 0.0
        return self.alpha * G**0.8 / L**0.2 * np.exp(-self.beta * rho_p * r / G)

    def total_rate(self, r0, G, d_char, x_head, propellant):
        L = d_char if self.length_scale == "hydraulic_diameter" else x_head
        if G <= 0.0 or L <= 0.0:
            return r0
        rho = propellant.density
        # f(r) = r - r0 - re(r) is increasing in r, f(r0) <= 0, so bracket [r0, r0 + re(r0)].
        upper = r0 + self.erosive_increment(r0, G, L, rho)
        if upper - r0 < 1e-15:
            return r0
        f = lambda r: r - r0 - self.erosive_increment(r, G, L, rho)  # noqa: E731
        if f(upper) <= 0.0:  # blowing term negligible: fixed point is the upper bound
            return upper
        return brentq(f, r0, upper, xtol=1e-12)


@dataclass(frozen=True)
class MukundaPaul:
    """Mukunda & Paul (1997) universal erosive-burning correlation."""

    g_threshold: float = 35.0
    coefficient: float = 0.023

    def eta(self, r0, G, d_char, propellant) -> float:
        if r0 <= 0.0 or G <= 0.0 or d_char <= 0.0:
            return 1.0
        rho = propellant.density
        g0 = G / (rho * r0)
        re0 = rho * r0 * d_char / propellant.mu_gas
        g = g0 * (re0 / 1000.0) ** -0.125
        if g <= self.g_threshold:
            return 1.0
        return 1.0 + self.coefficient * (g**0.8 - self.g_threshold**0.8)

    def total_rate(self, r0, G, d_char, x_head, propellant):
        return r0 * self.eta(r0, G, d_char, propellant)
