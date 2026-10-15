"""Propellant thermochemistry and steady burn-rate law.

The burn rate follows Saint Robert's (Vieille's) law with an exponential
temperature sensitivity referenced to a nominal grain temperature:

    r0(p, T_i) = a_ref * exp(sigma_p * (T_i - T_ref)) * (p / p_ref)**n

Using p_ref makes ``a_ref`` a burn rate in m/s at a stated pressure, which is how
burn-rate data are usually reported (e.g. 0.541 cm/s at 6.9 MPa for NAWC motor 13
in Willcox et al., 2007).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Propellant:
    """Lumped propellant and combustion-gas properties.

    Parameters
    ----------
    density : propellant density rho_p [kg/m^3]
    a_ref : burn rate at ``p_ref`` and ``T_ref`` [m/s]
    n : pressure exponent [-]
    c_star : characteristic velocity c* [m/s]
    gamma : ratio of specific heats of the combustion products [-]
    p_ref : reference pressure for ``a_ref`` [Pa]
    T_ref : reference grain temperature [K]
    sigma_p : temperature sensitivity of burn rate at constant pressure [1/K]
    c_star_efficiency : delivered fraction of theoretical c* [-]
    cp_gas, mu_gas, pr_gas : gas heat capacity [J/kg/K], viscosity [Pa s] and
        Prandtl number, used only by erosive-burning models
    """

    density: float
    a_ref: float
    n: float
    c_star: float
    gamma: float
    p_ref: float = 6.9e6
    T_ref: float = 294.15
    sigma_p: float = 0.0
    c_star_efficiency: float = 1.0
    cp_gas: float = 2000.0
    mu_gas: float = 8.0e-5
    pr_gas: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.n < 1.0:
            raise ValueError("pressure exponent n must satisfy 0 <= n < 1 for stable burning")
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed 1")

    @property
    def c_star_delivered(self) -> float:
        return self.c_star * self.c_star_efficiency

    @property
    def vandenkerckhove(self) -> float:
        """Gamma(gamma) = sqrt(g) * (2/(g+1))**((g+1)/(2(g-1)))."""
        g = self.gamma
        return np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))

    @property
    def RT0(self) -> float:
        """R*T0 of the chamber gas implied by c*: R*T0 = (c* Gamma)^2 [J/kg]."""
        return (self.c_star_delivered * self.vandenkerckhove) ** 2

    def a(self, T_grain: float | None = None) -> float:
        """Burn-rate coefficient (m/s at p_ref) at grain temperature ``T_grain`` [K]."""
        if T_grain is None:
            return self.a_ref
        return self.a_ref * np.exp(self.sigma_p * (T_grain - self.T_ref))

    def burn_rate(self, p, T_grain: float | None = None):
        """Non-erosive steady burn rate r0 [m/s] at chamber pressure ``p`` [Pa]."""
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        return self.a(T_grain) * (p / self.p_ref) ** self.n

    def with_(self, **changes) -> "Propellant":
        """Return a copy with some fields changed (used by the UQ driver)."""
        return replace(self, **changes)
