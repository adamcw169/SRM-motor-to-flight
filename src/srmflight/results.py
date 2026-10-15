"""Container for motor time histories and scalar performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class MotorResult:
    """Time histories from an internal-ballistics run (SI units)."""

    t: np.ndarray
    pressure: np.ndarray  # head-end chamber pressure [Pa]
    thrust: np.ndarray  # [N]
    mass_flow: np.ndarray  # nozzle mass flow [kg/s]
    web: np.ndarray  # burnt web (0D) or mean burnt web (1D) [m]
    burn_area: np.ndarray  # total burning area [m^2]
    throat_radius: np.ndarray  # [m]
    propellant_mass: float  # initial propellant mass [kg]
    extra: dict = field(default_factory=dict)

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressure))

    @property
    def total_impulse(self) -> float:
        return float(trapezoid(self.thrust, self.t))

    @property
    def expelled_mass(self) -> float:
        return float(trapezoid(self.mass_flow, self.t))

    @property
    def delivered_isp(self) -> float:
        """Total impulse / expelled mass [m/s] (effective exhaust velocity)."""
        m = self.expelled_mass
        return self.total_impulse / m if m > 0 else 0.0

    def action_time(self, fraction: float = 0.10) -> float:
        """Time between the first and last instants with thrust >= fraction * peak thrust."""
        above = np.nonzero(self.thrust >= fraction * np.max(self.thrust))[0]
        if above.size == 0:
            return 0.0
        return float(self.t[above[-1]] - self.t[above[0]])

    def summary(self) -> dict:
        return {
            "peak_pressure_MPa": self.peak_pressure / 1e6,
            "total_impulse_Ns": self.total_impulse,
            "action_time_s": self.action_time(),
            "delivered_isp_mps": self.delivered_isp,
            "mass_closure": self.expelled_mass / self.propellant_mass,
        }
