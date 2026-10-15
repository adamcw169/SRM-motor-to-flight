"""Lumped-parameter (0D) internal ballistics.

Chamber mass balance with an isothermal gas at the delivered c* (Willcox et al.
2007, Eq. 3, with the free-volume growth term retained):

    V/(R T0) dP/dt = A_b r (rho_p - P/(R T0)) - P A_t / c*
    dw/dt  = r = a (P/p_ref)^n
    dr_t/dt = k (P/p_ref_e)^m           (throat erosion)

Pressure is uniform, so erosive burning is excluded here; use the quasi-1D solver
for slender or port-restricted grains. Ignition is not modelled: the run starts
from the equilibrium pressure of the unburnt grain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..geometry.grain import Grain
from ..nozzle import Nozzle
from ..propellant import Propellant
from ..results import MotorResult


@dataclass
class ZeroDMotor:
    grain: Grain
    propellant: Propellant
    nozzle: Nozzle
    T_grain: float | None = None  # grain bulk temperature [K]; None = reference
    ambient_pressure: float = 101_325.0
    table_points: int = 4001  # web resolution of the cached A_b(w), V(w) tables

    def __post_init__(self) -> None:
        # Grain geometry is tabulated once; the ODE right-hand side then only interpolates.
        wmax = self.grain.web_max
        self._w = np.linspace(0.0, wmax * (1.0 - 1e-12), self.table_points)
        self._ab = np.asarray(self.grain.burn_area(self._w), dtype=float)
        self._v = np.asarray(self.grain.free_volume(self._w), dtype=float)
        self._v_end = float(self.grain.free_volume(wmax))

    # ------------------------------------------------------------- algebra
    def equilibrium_pressure(self, w: float, throat_radius: float | None = None) -> float:
        """Steady chamber pressure at web ``w`` (dP/dt = 0, gas storage included)."""
        rt = 0.5 * self.nozzle.throat_diameter if throat_radius is None else throat_radius
        at = np.pi * rt**2
        ab = float(self.grain.burn_area(w))
        if ab <= 0.0:
            return 0.0
        p = self.propellant
        rt0, cs = p.RT0, p.c_star_delivered

        def f(P):
            return ab * p.burn_rate(P, self.T_grain) * (p.density - P / rt0) - P * at / cs

        return brentq(f, 1.0, 1e9)

    def equilibrium_pressure_closed_form(self, w: float) -> float:
        """P = (rho_p a A_b c* / (A_t p_ref^n))^(1/(1-n)); neglects gas storage."""
        p = self.propellant
        ab = float(self.grain.burn_area(w))
        k = p.density * p.a(self.T_grain) * ab * p.c_star_delivered / (
            self.nozzle.throat_area * p.p_ref**p.n
        )
        return k ** (1.0 / (1.0 - p.n))

    # ------------------------------------------------------------- dynamics
    def _rhs(self, t, y, burning: bool):
        P, w, rt = y
        prop, noz = self.propellant, self.nozzle
        P = max(P, 0.0)
        rt0, cs = prop.RT0, prop.c_star_delivered
        at = np.pi * rt**2
        if burning:
            r = prop.a(self.T_grain) * (P / prop.p_ref) ** prop.n
            ab = np.interp(w, self._w, self._ab)
            V = np.interp(w, self._w, self._v)
        else:
            r, ab, V = 0.0, 0.0, self._v_end
        dP = rt0 / V * (ab * r * (prop.density - P / rt0) - P * at / cs)
        return [dP, r, noz.erosion.rate(P)]

    def run(self, t_max: float = 30.0, n_out: int = 2000, rtol: float = 1e-8) -> MotorResult:
        """Integrate from the equilibrium start to tail-off (P below 1.5 x ambient)."""
        rt0 = 0.5 * self.nozzle.throat_diameter
        y0 = [self.equilibrium_pressure(0.0), 0.0, rt0]
        web_max = self.grain.web_max

        burnout = lambda t, y, b: y[1] - web_max  # noqa: E731
        burnout.terminal, burnout.direction = True, 1
        sol1 = solve_ivp(self._rhs, (0.0, t_max), y0, args=(True,), method="LSODA",
                         events=burnout, rtol=rtol, atol=[1.0, 1e-10, 1e-10], dense_output=True)
        t_b = sol1.t[-1]

        pa = self.ambient_pressure
        blowdown = lambda t, y, b: y[0] - 1.5 * pa  # noqa: E731
        blowdown.terminal, blowdown.direction = True, -1
        y1 = sol1.y[:, -1].copy()
        y1[1] = web_max
        sol2 = solve_ivp(self._rhs, (t_b, t_b + 5.0), y1, args=(False,), method="LSODA",
                         events=blowdown, rtol=rtol, atol=[1.0, 1e-10, 1e-10], dense_output=True)

        n1 = int(n_out * 0.9)
        t1 = np.linspace(0.0, t_b, n1)
        t2 = np.linspace(t_b, sol2.t[-1], n_out - n1 + 1)[1:]
        Y = np.hstack([sol1.sol(t1), sol2.sol(t2)])
        t = np.concatenate([t1, t2])
        P, w, rt = Y[0].clip(min=0.0), np.minimum(Y[1], web_max), Y[2]

        prop, noz = self.propellant, self.nozzle
        mdot = P * np.pi * rt**2 / prop.c_star_delivered
        F = noz.thrust(P, rt, prop.gamma, pa)
        Ab = np.where(t <= t_b, self.grain.burn_area(w), 0.0)
        m0 = float(self.grain.propellant_volume(0.0)) * prop.density
        return MotorResult(t, P, F, mdot, w, Ab, rt, m0, extra={"burnout_time": t_b})
