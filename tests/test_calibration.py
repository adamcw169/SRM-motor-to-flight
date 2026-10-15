"""Thrust-curve calibration: parser and parameter recovery on a synthetic certificate."""

import numpy as np
import pytest

from srmflight import Nozzle, Propellant, ThroatErosion
from srmflight.ballistics import ZeroDMotor
from srmflight.calibration import EngCurve, fit_thrust_curve, read_eng
from srmflight.geometry import CircularPort, Grain

GRAIN = Grain.bates(CircularPort(0.066, 0.030), 0.120, 5)
N_TRUE = 0.40


def motor(p):
    prop = Propellant(density=1520.0, a_ref=p["a_ref"], n=N_TRUE, c_star=p["c_star"], gamma=1.2)
    noz = Nozzle(0.0215, 0.0495, efficiency=0.95,
                 erosion=ThroatErosion(k=p["k"] * 1e-3, m=1.0, p_ref=6.9e6))
    return ZeroDMotor(GRAIN, prop, noz).run(n_out=600)


def test_read_eng(tmp_path):
    f = tmp_path / "m.eng"
    f.write_text("; comment\nX100-T 75 621 0 2.475 4.323 CTI\n0.1 100\n1.0 120\n2.0 0\n")
    c = read_eng(f)
    assert c.name == "X100-T" and c.propellant_mass == 2.475
    assert c.total_impulse == pytest.approx(0.5 * 0.1 * 100 + 0.5 * 0.9 * 220 + 0.5 * 1.0 * 120)


def test_fit_recovers_known_motor():
    truth = {"a_ref": 0.0081, "c_star": 1434.0, "k": 0.9}
    res = motor(truth)
    t = np.linspace(0.005, res.t[-1], 400)
    curve = EngCurve("synthetic", 75, 621, res.propellant_mass, 4.3, t,
                     np.interp(t, res.t, res.thrust))
    fit = fit_thrust_curve(motor, curve, {"a_ref": 0.007, "c_star": 1300.0, "k": 0.5},
                           {"a_ref": (0.002, 0.03), "c_star": (900, 1800), "k": (0, 3)},
                           window=(0.25, 3.0))
    for key, val in truth.items():
        assert fit.params[key] == pytest.approx(val, rel=2e-2)
    assert fit.rms_error < 5e-3
