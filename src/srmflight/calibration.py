"""Calibrate an internal-ballistics model to a measured (certified) thrust curve.

Commercial motors publish their grain geometry and static-test thrust, but not their
propellant's burn-rate law or c*. With the thrust efficiency η_F fixed, the remaining
unknowns are identifiable from a thrust trace:

    total impulse   I ≈ η_F C_F c* m_p        ->  c*
    thrust level    F ≈ η_F C_F p_c A_t       ->  chamber pressure
    burn time       t_b ≈ w / (a p_c^n)       ->  a
    curve shape     dF/dt via A_b(w), n       ->  n

``fit_thrust_curve`` solves for (a_ref, n, c*) by least squares on the thrust trace over the
quasi-steady part of the burn, plus a total-impulse constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares

from .results import MotorResult


@dataclass
class EngCurve:
    """RASP .eng thrust curve (time [s], thrust [N]) with its header."""

    name: str
    diameter_mm: float
    length_mm: float
    propellant_mass: float  # kg
    total_mass: float  # kg
    t: np.ndarray
    thrust: np.ndarray

    @property
    def total_impulse(self) -> float:
        t, F = self._closed()
        return float(trapezoid(F, t))

    def _closed(self):
        t, F = np.concatenate([[0.0], self.t]), np.concatenate([[0.0], self.thrust])
        return t, F

    def thrust_at(self, t):
        t0, F0 = self._closed()
        return np.interp(t, t0, F0, right=0.0)


def read_eng(path: str | Path) -> EngCurve:
    rows = [ln.split(";")[0].strip() for ln in Path(path).read_text().splitlines()]
    rows = [r for r in rows if r]
    head = rows[0].split()
    data = np.array([[float(v) for v in r.split()[:2]] for r in rows[1:]])
    return EngCurve(head[0], float(head[1]), float(head[2]), float(head[4]), float(head[5]),
                    data[:, 0], data[:, 1])


@dataclass
class FitResult:
    params: dict
    result: MotorResult
    rms_error: float  # RMS thrust error over the fit window / mean thrust
    impulse_error: float  # relative
    window: tuple[float, float]


def fit_thrust_curve(
    run: Callable[[dict], MotorResult],
    curve: EngCurve,
    x0: dict,
    bounds: dict,
    window: tuple[float, float] = (0.1, 3.0),
    impulse_weight: float = 5.0,
    n_samples: int = 120,
) -> FitResult:
    """Least-squares fit of ``run(params) -> MotorResult`` to ``curve``.

    ``x0`` and ``bounds`` share keys; bounds are (low, high) tuples.
    """
    names = list(x0)
    t_fit = np.linspace(*window, n_samples)
    F_ref = curve.thrust_at(t_fit)
    I_ref = curve.total_impulse
    scale = float(np.mean(F_ref))

    def residual(x):
        res = run(dict(zip(names, x)))
        F = np.interp(t_fit, res.t, res.thrust, right=0.0)
        return np.concatenate([(F - F_ref) / scale,
                               [impulse_weight * (res.total_impulse - I_ref) / I_ref]])

    lo = [bounds[k][0] for k in names]
    hi = [bounds[k][1] for k in names]
    sol = least_squares(residual, [x0[k] for k in names], bounds=(lo, hi), x_scale="jac",
                        diff_step=1e-4)
    params = dict(zip(names, sol.x))
    res = run(params)
    F = np.interp(t_fit, res.t, res.thrust, right=0.0)
    rms = float(np.sqrt(np.mean((F - F_ref) ** 2)) / scale)
    return FitResult(params, res, rms, (res.total_impulse - I_ref) / I_ref, window)
