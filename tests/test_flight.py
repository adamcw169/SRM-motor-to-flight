"""Flight-coupling helpers that do not need RocketPy installed."""

import numpy as np

from srmflight.ballistics import ZeroDMotor
from srmflight.flight import exhaust_velocity_error, thrust_curve, write_thrust_csv
from srmflight.geometry import CircularPort, Grain


def test_thrust_curve_and_csv(tmp_path, propellant, nozzle):
    res = ZeroDMotor(Grain.single(CircularPort(0.1, 0.04), 0.5), propellant, nozzle).run()
    curve = thrust_curve(res)
    assert curve[0, 0] == 0.0 and curve[-1, 1] == 0.0
    resampled_impulse = np.trapezoid(curve[:, 1], curve[:, 0])
    assert abs(resampled_impulse / res.total_impulse - 1.0) < 1e-2
    path = write_thrust_csv(res, tmp_path / "thrust.csv")
    assert np.loadtxt(path, delimiter=",").shape == curve.shape


def test_constant_exhaust_velocity_error_is_small_but_nonzero(propellant, nozzle):
    """Cf varies with pressure, so RocketPy's constant-v_e mass flow is only approximate."""
    res = ZeroDMotor(Grain.single(CircularPort(0.1, 0.04), 0.5), propellant, nozzle).run()
    err = exhaust_velocity_error(res)
    assert 0.0 < err < 0.05
