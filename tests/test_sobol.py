"""Sobol estimator verification on the Ishigami function (analytic indices)."""

import numpy as np
from scipy import stats

from srmflight.uq import run_sobol

A, B = 7.0, 0.1
S1_EXACT = [0.3139, 0.4424, 0.0]
ST_EXACT = [0.5576, 0.4424, 0.2437]


def ishigami(p):
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return {"y": np.sin(x1) + A * np.sin(x2) ** 2 + B * x3**4 * np.sin(x1)}


def test_ishigami_indices():
    u = stats.uniform(-np.pi, 2 * np.pi)
    res = run_sobol(ishigami, {"x1": u, "x2": u, "x3": u}, n_base=2**12, seed=3)["y"]
    np.testing.assert_allclose(res.S1, S1_EXACT, atol=0.03)
    np.testing.assert_allclose(res.ST, ST_EXACT, atol=0.03)
    assert np.all(res.ST_ci[:, 0] <= res.ST + 1e-12)
