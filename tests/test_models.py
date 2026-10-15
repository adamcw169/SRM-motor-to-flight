"""Burn-rate, erosive-burning and nozzle model checks."""

import numpy as np
import pytest

from srmflight import LenoirRobillard, MukundaPaul, Propellant, thrust_coefficient
from srmflight.nozzle import area_ratio, exit_mach


def test_burn_rate_reference_point(propellant):
    assert propellant.burn_rate(propellant.p_ref) == pytest.approx(propellant.a_ref)


def test_temperature_sensitivity(propellant):
    dT = 40.0
    ratio = propellant.burn_rate(7e6, propellant.T_ref + dT) / propellant.burn_rate(7e6)
    assert np.log(ratio) / dT == pytest.approx(propellant.sigma_p)


def test_rejects_unstable_exponent():
    with pytest.raises(ValueError):
        Propellant(density=1700, a_ref=0.01, n=1.05, c_star=1500, gamma=1.2)


def test_lenoir_robillard_fixed_point(propellant):
    lr = LenoirRobillard(alpha=2e-5)
    r0, G, L = 0.006, 1500.0, 0.03
    r = lr.total_rate(r0, G, L, 0.5, propellant)
    assert r > r0
    assert r == pytest.approx(r0 + lr.erosive_increment(r, G, L, propellant.density), rel=1e-9)


def test_lenoir_robillard_zero_flux(propellant):
    assert LenoirRobillard(alpha=2e-5).total_rate(0.006, 0.0, 0.03, 0.0, propellant) == 0.006


def test_lenoir_robillard_alpha_grain_temperature_coupling():
    args = (2000, 8e-5, 0.5, 1750, 1250, 3000, 1000)
    cold = LenoirRobillard.from_properties(*args, T_grain=233.15).alpha
    hot = LenoirRobillard.from_properties(*args, T_grain=333.15).alpha
    assert hot / cold == pytest.approx((1000 - 233.15) / (1000 - 333.15))


def test_mukunda_paul_threshold(propellant):
    mp = MukundaPaul()
    assert mp.eta(0.006, 10.0, 0.03, propellant) == 1.0
    assert mp.eta(0.006, 3000.0, 0.03, propellant) > 1.0


def test_isentropic_area_ratio():
    assert area_ratio(1.0, 1.4) == pytest.approx(1.0)
    assert area_ratio(2.0, 1.4) == pytest.approx(1.6875, rel=1e-4)
    assert exit_mach(1.6875, 1.4) == pytest.approx(2.0, rel=1e-6)


def test_thrust_coefficient_vacuum_limit():
    """Cf rises with expansion ratio and tends to the vacuum limit."""
    g = 1.2
    cf = [thrust_coefficient(g, eps, 7e6, 0.0) for eps in (4, 16, 64)]
    assert cf[0] < cf[1] < cf[2]
    cf_inf = np.sqrt(2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)))
    assert cf[2] < cf_inf
