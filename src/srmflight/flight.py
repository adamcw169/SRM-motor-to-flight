"""Coupling to RocketPy 6-DOF flight (optional dependency: ``pip install srmflight[flight]``).

RocketPy's motor classes take a thrust curve and derive propellant mass flow from
a constant exhaust velocity, v_e = I_tot / m_p (Ceotto et al. 2021, Eq. 1). The
ballistics model here predicts the real thrust *and* mass-flow histories, so
``exhaust_velocity_error`` reports how far that assumption is from the physics.

The RocketPy API has changed between releases. ``to_rocketpy_motor`` targets the
``GenericMotor`` class of RocketPy 1.x; check the signature against the installed
version (``help(rocketpy.GenericMotor)``) before relying on it.
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


def to_rocketpy_motor(
    result: MotorResult,
    chamber_radius: float,
    chamber_height: float,
    chamber_position: float,
    nozzle_radius: float,
    dry_mass: float,
    dry_inertia: tuple[float, float, float],
    center_of_dry_mass_position: float,
    nozzle_position: float = 0.0,
):
    """Build a RocketPy ``GenericMotor`` from a ballistics result (RocketPy 1.x)."""
    try:
        from rocketpy import GenericMotor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the flight extra: pip install 'srmflight[flight]'") from exc

    curve = thrust_curve(result)
    return GenericMotor(
        thrust_source=curve,
        burn_time=float(curve[-1, 0]),
        chamber_radius=chamber_radius,
        chamber_height=chamber_height,
        chamber_position=chamber_position,
        propellant_initial_mass=result.expelled_mass,
        nozzle_radius=nozzle_radius,
        dry_mass=dry_mass,
        dry_inertia=dry_inertia,
        center_of_dry_mass_position=center_of_dry_mass_position,
        nozzle_position=nozzle_position,
    )
