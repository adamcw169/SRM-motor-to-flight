"""Quasi-steady, quasi-1D internal ballistics with erosive burning.

At each time step the port flow is solved as a steady 1D inviscid flow with
normal mass injection (Shapiro, Vol. I; Willcox et al. 2007 use the unsteady
form). Marching from the head end (M = 0) along the axial stations:

    mdot_{j+1} = mdot_j + rho_p r_j S_j dz                      (mass)
    (pA + mdot u)_{j+1} = (pA + mdot u)_j + p_j (A_{j+1} - A_j)  (axial momentum)
    T0 = const,  T = T0 / (1 + (g-1)/2 M^2)                    (energy)

The head-end pressure is found by root finding so that the aft-end stagnation
pressure chokes the nozzle with exactly the generated mass flow:

    mdot_N = p0_N A_t / c*

Each station regresses with its own local rate r_j = r0(p_j) augmented by the
erosive model at the local mass flux G_j = mdot_j / A_j and hydraulic diameter
D_h = 4 A_j / P_wet,j. Gas storage and ignition transients are neglected (valid
once the chamber has filled; Willcox et al. show dynamic burning decays within
~0.15 s for NAWC motor 13).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from ..erosive import NoErosion
from ..geometry.grain import AxialGrain
from ..nozzle import Nozzle
from ..propellant import Propellant
from ..results import MotorResult


class PortChokedError(RuntimeError):
    """Port flow reached Mach 1 before the nozzle (grain design not admissible)."""


def _mach_from_flux_ratio(q: float, gamma: float, RT0: float) -> float:
    """Subsonic M solving mdot/F = M sqrt(g/RT0) sqrt(1+(g-1)/2 M^2) / (1 + g M^2)."""
    g = gamma

    def h(M):
        return M * np.sqrt(g / RT0) * np.sqrt(1.0 + 0.5 * (g - 1.0) * M**2) / (1.0 + g * M**2)

    if q <= 0.0:
        return 0.0
    if q >= h(1.0):
        raise PortChokedError("port Mach number reached 1")
    return brentq(lambda M: h(M) - q, 0.0, 1.0, xtol=1e-12)


@dataclass
class FlowSolution:
    p: np.ndarray
    mach: np.ndarray
    G: np.ndarray
    r: np.ndarray
    mdot: np.ndarray
    p0_aft: float


@dataclass
class QuasiOneDMotor:
    grain: AxialGrain
    propellant: Propellant
    nozzle: Nozzle
    erosive: object = field(default_factory=NoErosion)
    T_grain: float | None = None
    ambient_pressure: float = 101_325.0

    # ------------------------------------------------------------ steady flow
    def march(self, p_head: float, webs: np.ndarray) -> FlowSolution:
        prop, g = self.propellant, self.propellant.gamma
        RT0 = prop.RT0
        S, A, Pw = self.grain.evaluate(webs)
        dz = self.grain.station_length
        n = self.grain.n
        p, M, G, r, mdot = (np.zeros(n) for _ in range(5))
        md, F = 0.0, p_head * A[0]
        for j in range(n):
            if j > 0:
                F += p[j - 1] * (A[j] - A[j - 1])
            Mj = _mach_from_flux_ratio(md / F, g, RT0) if md > 0 else 0.0
            pj = F / (A[j] * (1.0 + g * Mj**2))
            Gj = md / A[j]
            r0 = float(prop.burn_rate(pj, self.T_grain))
            dh = 4.0 * A[j] / max(Pw[j], 1e-12)
            rj = self.erosive.total_rate(r0, Gj, dh, self.grain.z[j], prop) if S[j] > 0 else 0.0
            p[j], M[j], G[j], r[j], mdot[j] = pj, Mj, Gj, rj, md
            dm = prop.density * rj * S[j] * dz
            # mass enters with zero axial momentum: F is unchanged by injection
            md += dm
        # state after the last station (aft end, before the nozzle)
        M_aft = _mach_from_flux_ratio(md / F, g, RT0) if md > 0 else 0.0
        p_aft = F / (A[-1] * (1.0 + g * M_aft**2))
        p0_aft = p_aft * (1.0 + 0.5 * (g - 1.0) * M_aft**2) ** (g / (g - 1.0))
        mdot[-1] = md
        return FlowSolution(p, M, G, r, mdot, p0_aft)

    def solve_flow(self, webs: np.ndarray, throat_radius: float) -> FlowSolution | None:
        at = np.pi * throat_radius**2
        cs = self.propellant.c_star_delivered

        def residual(ph):
            try:
                sol = self.march(ph, webs)
            except PortChokedError:
                # too little head-end pressure to pass the generated mass: raise ph
                return 1e6
            return sol.mdot[-1] - sol.p0_aft * at / cs

        lo, hi = 1.05 * self.ambient_pressure, 5e6
        if residual(lo) <= 0.0:
            return None  # remaining sliver cannot sustain a choked nozzle: burnout
        while residual(hi) > 0.0:  # generated mass still exceeds nozzle capacity
            hi *= 2.0
            if hi > 1e9:
                raise RuntimeError("no pressure balance below 1 GPa")
        ph = brentq(residual, lo, hi, rtol=1e-10)
        return self.march(ph, webs)

    # ------------------------------------------------------------ time march
    def run(self, dw_fraction: float = 1 / 400, t_max: float = 60.0) -> MotorResult:
        prop, noz = self.propellant, self.nozzle
        webs = np.zeros(self.grain.n)
        wmax = self.grain.web_max()
        rt = 0.5 * noz.throat_diameter
        dw_target = dw_fraction * float(np.max(wmax))
        m0 = self.grain.propellant_volume(webs) * prop.density
        rec = {k: [] for k in ("t", "p", "F", "mdot", "w", "Ab", "rt", "p_aft")}
        profiles = []
        t = 0.0
        while t < t_max:
            active = webs < wmax
            if not np.any(active):
                break
            sol = self.solve_flow(webs, rt)
            if sol is None:
                break
            S, _, _ = self.grain.evaluate(webs)
            rmax = float(np.max(sol.r[active]))
            # step so that no station overshoots its burnout web (exact burnout times)
            remaining = (wmax - webs)[active] / np.maximum(sol.r[active], 1e-300)
            dt = min(dw_target / rmax, float(np.min(remaining)))
            dt = max(dt, 1e-3 * dw_target / rmax)
            F = noz.thrust(sol.p0_aft, rt, prop.gamma, self.ambient_pressure)
            for k, v in zip(rec, (t, sol.p[0], F, sol.mdot[-1], float(np.mean(webs)),
                                  float(np.sum(S) * self.grain.station_length), rt, sol.p0_aft)):
                rec[k].append(v)
            profiles.append((t, sol.p.copy(), sol.r.copy(), webs.copy()))
            webs = np.minimum(webs + sol.r * dt, wmax)
            webs[wmax - webs < 1e-6 * dw_target] = wmax[wmax - webs < 1e-6 * dw_target]
            rt += noz.erosion.rate(sol.p0_aft) * dt
            t += dt
        self._append_blowdown(rec, t, rt)
        arr = {k: np.asarray(v) for k, v in rec.items()}
        return MotorResult(arr["t"], arr["p"], arr["F"], arr["mdot"], arr["w"], arr["Ab"],
                           arr["rt"], m0, extra={"p_aft": arr["p_aft"], "profiles": profiles})

    def _append_blowdown(self, rec, t_b, rt, n=200):
        """Isothermal exponential tail-off of the empty chamber (no burning surface)."""
        prop = self.propellant
        V = self.grain.free_volume(self.grain.web_max())
        at = np.pi * rt**2
        tau = V * prop.c_star_delivered / (prop.RT0 * at)
        p_start = rec["p_aft"][-1] if rec["p_aft"] else 0.0
        pa = self.ambient_pressure
        if p_start <= 1.5 * pa:
            return
        t_end = tau * np.log(p_start / (1.5 * pa))
        for s in np.linspace(0.0, t_end, n)[1:]:
            p = p_start * np.exp(-s / tau)
            rec["t"].append(t_b + s)
            rec["p"].append(p)
            rec["p_aft"].append(p)
            rec["F"].append(self.nozzle.thrust(p, rt, prop.gamma, pa))
            rec["mdot"].append(p * at / prop.c_star_delivered)
            rec["w"].append(rec["w"][-1])
            rec["Ab"].append(0.0)
            rec["rt"].append(rt)
