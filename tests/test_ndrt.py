"""NDRT 2020 case: motor consistency (always) and flight validation (needs RocketPy + data)."""

from pathlib import Path

import numpy as np
import pytest

from srmflight.cases import ndrt2020 as nd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/ndrt/data"
CAL = ROOT / "validation/ndrt/calibration.json"


def test_loaded_mass_matches_certificate():
    assert float(nd.GRAIN.propellant_volume(0.0)) * nd.DENSITY == pytest.approx(2.475)


def test_calibrated_motor_reproduces_certified_impulse():
    cal = nd.Calibration.load(CAL)
    for n in (cal.n[0], 0.4, cal.n[-1]):
        res = nd.l1395(**cal.at(n))
        assert res.total_impulse == pytest.approx(4895.05, rel=1e-2)
        assert res.summary()["mass_closure"] == pytest.approx(1.0, abs=2e-3)
    assert np.ptp(cal.c_star) / np.mean(cal.c_star) < 5e-3  # c* identified at every n


def test_hot_grain_burns_faster():
    p = nd.Calibration.load(CAL).at(0.4)
    cold, hot = nd.l1395(**p, T_grain=228.15), nd.l1395(**p, T_grain=333.15)
    assert hot.peak_pressure > cold.peak_pressure
    assert hot.action_time() < cold.action_time()


@pytest.mark.skipif(not (DATA / "ndrt_2020_flight_data.csv").exists(),
                    reason="run validation/ndrt/fetch_data.py first")
def test_flight_apogee_against_measurement():
    pytest.importorskip("rocketpy")
    cal = nd.Calibration.load(CAL)
    env = nd.environment(DATA)
    _, z = nd.measured_flight(DATA)
    fl = nd.fly(nd.motor_from_result(nd.l1395(**cal.at(0.4))), env)
    # same 1.5 % tolerance as RocketPy's own NDRT acceptance test
    assert abs(fl.apogee - nd.ELEVATION - z.max()) / z.max() < 0.015
