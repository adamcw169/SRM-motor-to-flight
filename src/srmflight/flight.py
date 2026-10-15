"""Coupling to RocketPy 6-DOF flight (optional dependency: ``pip install srmflight[flight]``).

RocketPy's motor classes take a thrust curve and derive propellant mass flow from
a constant exhaust velocity, v_e = I_tot / m_p (Ceotto et al. 2021, Eq. 1). The
ballistics model here predicts the real thrust *and* mass-flow histories, so
``exhaust_velocity_error`` reports how far that assumption is from the physics.

A worked RocketPy set-up (``SolidMotor`` with real grain geometry, airframe, ERA5
atmosphere) is in ``srmflight.cases.ndrt2020``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .results import MotorResult


def thrust_curve(result: MotorResult, n: int = 400) -> np.ndarray:
    """(t, F) table resampled to ``n`` points, starting at t = 0 and ending at F = 0."""
    t = np.linspace(result.t[0], result.t[-1], n)
    F = np.interp(t, result.t, result.thrust)
    F[-1] = 0.0
    return np.column_stack([t - t[0], F])


def write_thrust_csv(result: MotorResult, path: str | Path) -> Path:
    """Two-column CSV (time [s], thrust [N]) accepted by RocketPy as ``thrust_source``."""
    path = Path(path)
    np.savetxt(path, thrust_curve(result), delimiter=",", fmt="%.6e")
    return path


def exhaust_velocity_error(result: MotorResult) -> float:
    """Max relative error in expelled-mass history implied by RocketPy's constant v_e."""
    ve = result.total_impulse / result.expelled_mass
    m_true = cumulative_trapezoid(result.mass_flow, result.t, initial=0.0)
    m_const = cumulative_trapezoid(result.thrust / ve, result.t, initial=0.0)
    return float(np.max(np.abs(m_const - m_true)) / result.expelled_mass)
