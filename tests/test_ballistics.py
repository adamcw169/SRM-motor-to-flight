"""0D and quasi-1D internal ballistics verification."""

import numpy as np
import pytest

from srmflight import LenoirRobillard, Nozzle, NoErosion, ThroatErosion
from srmflight.ballistics import PortChokedError, QuasiOneDMotor, ZeroDMotor
from srmflight.geometry import AxialGrain, CircularPort, Grain

SECTION = CircularPort(0.1, 0.04)


def test_equilibrium_pressure_closed_form(propellant, nozzle):
    m = ZeroDMotor(Grain.single(SECTION, 0.5), propellant, nozzle)
    # closed form neglects gas storage (rho_g / rho_p ~ 0.5 %)
    assert m.equilibrium_pressure(0.0) == pytest.approx(
        m.equilibrium_pressure_closed_form(0.0), rel=1e-2)


def test_tube_grain_burn_time_analytic(propellant, nozzle):
    """Progressive tube, inhibited ends: dw/dt = c (r_p + w)^m with m = n/(1-n)."""
    m = ZeroDMotor(Grain.single(SECTION, 0.5), propellant, nozzle)
    res = m.run()
    n, rp, R = propellant.n, 0.02, 0.05
    k = n / (1 - n)
    c = propellant.a() * (m.equilibrium_pressure_closed_form(0.0) / propellant.p_ref) ** n / rp**k
    t_burn = (R ** (1 - k) - rp ** (1 - k)) / ((1 - k) * c)
    assert res.extra["burnout_time"] == pytest.approx(t_burn, rel=1e-2)


def test_mass_closure_0d(propellant, nozzle):
    res = ZeroDMotor(Grain.single(SECTION, 0.5), propellant, nozzle).run()
    assert res.expelled_mass / res.propellant_mass == pytest.approx(1.0, abs=2e-3)


def test_hot_grain_raises_pressure(propellant, nozzle):
    g = Grain.single(SECTION, 0.5)
    cold = ZeroDMotor(g, propellant, nozzle, T_grain=233.15).run()
    hot = ZeroDMotor(g, propellant, nozzle, T_grain=333.15).run()
    assert hot.peak_pressure > cold.peak_pressure
    assert hot.action_time() < cold.action_time()


def test_throat_erosion_lowers_late_pressure(propellant):
    g = Grain.single(SECTION, 0.5)
    clean = ZeroDMotor(g, propellant, Nozzle(0.02, 0.05)).run()
    eroded = ZeroDMotor(g, propellant, Nozzle(0.02, 0.05, erosion=ThroatErosion(k=2e-5))).run()
    assert eroded.throat_radius[-1] > clean.throat_radius[-1]
    assert eroded.peak_pressure < clean.peak_pressure


def test_quasi1d_matches_0d_for_stubby_grain(propellant, nozzle):
    """Low L/D and large port: axial pressure drop small, 1D -> 0D."""
    sec = CircularPort(0.1, 0.06)
    zero = ZeroDMotor(Grain.single(sec, 0.15), propellant, nozzle).run()
    one = QuasiOneDMotor(AxialGrain.uniform(sec, 0.15, 10), propellant, nozzle).run()
    assert one.pressure[0] == pytest.approx(zero.pressure[0], rel=1e-2)
    assert one.total_impulse == pytest.approx(zero.total_impulse, rel=1e-2)


def test_quasi1d_mass_closure_converges(propellant, nozzle):
    grain = AxialGrain.uniform(SECTION, 0.5, 20)
    errs = [abs(QuasiOneDMotor(grain, propellant, nozzle).run(dw_fraction=f).summary()
                ["mass_closure"] - 1.0) for f in (1 / 100, 1 / 400)]
    assert errs[1] < errs[0]
    assert errs[1] < 1e-3


def test_erosive_burning_peaks_at_aft_end(propellant):
    lr = LenoirRobillard.from_properties(2000, 8e-5, 0.5, 1750, 1250, 3000, 1000, 294.15)
    grain = AxialGrain.uniform(CircularPort(0.1, 0.035), 1.6, 40)
    noz = Nozzle(0.025, 0.06)
    flow_plain = QuasiOneDMotor(grain, propellant, noz, NoErosion()).solve_flow(
        np.zeros(grain.n), 0.0125)
    flow_ero = QuasiOneDMotor(grain, propellant, noz, lr).solve_flow(np.zeros(grain.n), 0.0125)
    assert flow_plain.r[-1] < flow_plain.r[0]  # pressure drop slows the aft end
    assert flow_ero.r[-1] > 1.5 * flow_ero.r[0]  # erosion reverses it
    assert flow_ero.p[0] > flow_plain.p[0]


def test_port_choking_detected(propellant):
    grain = AxialGrain.uniform(CircularPort(0.1, 0.012), 2.0, 40)
    motor = QuasiOneDMotor(grain, propellant, Nozzle(0.02, 0.05))
    with pytest.raises((PortChokedError, RuntimeError, ValueError)):
        motor.march(5e6, np.zeros(grain.n))
