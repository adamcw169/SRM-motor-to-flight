"""Print the verification table in docs/VERIFICATION.md (each row is also a unit test).

Run:  python examples/00_verification_report.py
"""

import numpy as np
from scipy import stats

from srmflight import Nozzle, Propellant
from srmflight.ballistics import QuasiOneDMotor, ZeroDMotor
from srmflight.geometry import (AxialGrain, CircularPort, DistanceFieldSection, Grain,
                                circle_polygon, square_polygon, star_polygon)
from srmflight.uq import run_sobol

prop = Propellant(density=1750, a_ref=0.007, n=0.35, c_star=1500, gamma=1.2)
noz = Nozzle(0.02, 0.05)
OD = 0.1
rows = []

# 1. distance-field tube vs closed form
num, ref = DistanceFieldSection(OD, circle_polygon(0.02)), CircularPort(OD, 0.04)
w = np.linspace(0, 0.98 * ref.web_max, 60)
err = np.max(np.abs(num.perimeter(w) / ref.perimeter(w) - 1))
rows.append(("Tube burnback, distance field vs exact perimeter", "max rel. error", err, 1e-3))

# 2. convex offset (Steiner formula)
sq = DistanceFieldSection(OD, square_polygon(0.03))
w = np.linspace(0.001, 0.015, 30)
err = np.max(np.abs(sq.perimeter(w) / (0.12 + 2 * np.pi * w) - 1))
rows.append(("Square port, offset perimeter S0 + 2πw", "max rel. error", err, 2e-3))

# 3. star grid convergence
star = star_polygon(6, 0.03, 0.015)
pref = DistanceFieldSection(OD, star, resolution=1001).perimeter(0.01)
e401 = abs(DistanceFieldSection(OD, star, resolution=401).perimeter(0.01) / pref - 1)
rows.append(("Star perimeter, 401 vs 1001 grid", "rel. difference", e401, 1e-3))

# 4. 0D burn time vs analytic progressive tube
m = ZeroDMotor(Grain.single(CircularPort(OD, 0.04), 0.5), prop, noz)
res = m.run()
n, rp, R = prop.n, 0.02, 0.05
k = n / (1 - n)
c = prop.a() * (m.equilibrium_pressure_closed_form(0.0) / prop.p_ref) ** n / rp**k
tb = (R ** (1 - k) - rp ** (1 - k)) / ((1 - k) * c)
rows.append(("0D burn time vs closed-form tube solution", "rel. error",
             abs(res.extra["burnout_time"] / tb - 1), 1e-2))
rows.append(("0D mass closure (expelled / loaded)", "deviation from 1",
             abs(res.expelled_mass / res.propellant_mass - 1), 2e-3))

# 5. quasi-1D consistency
one = QuasiOneDMotor(AxialGrain.uniform(CircularPort(OD, 0.04), 0.5, 20), prop, noz).run(
    dw_fraction=1 / 400)
rows.append(("Quasi-1D mass closure, Δw = web/400", "deviation from 1",
             abs(one.expelled_mass / one.propellant_mass - 1), 1e-3))
sec = CircularPort(OD, 0.06)
z = ZeroDMotor(Grain.single(sec, 0.15), prop, noz).run()
o = QuasiOneDMotor(AxialGrain.uniform(sec, 0.15, 10), prop, noz).run()
rows.append(("Quasi-1D → 0D limit (L/D = 2.5), total impulse", "rel. difference",
             abs(o.total_impulse / z.total_impulse - 1), 1e-2))

# 6. Sobol estimator on Ishigami
u = stats.uniform(-np.pi, 2 * np.pi)
f = lambda p: {"y": np.sin(p["x1"]) + 7 * np.sin(p["x2"]) ** 2  # noqa: E731
               + 0.1 * p["x3"] ** 4 * np.sin(p["x1"])}
s = run_sobol(f, {"x1": u, "x2": u, "x3": u}, n_base=2**12, seed=3)["y"]
err = max(np.max(np.abs(s.S1 - [0.3139, 0.4424, 0.0])),
          np.max(np.abs(s.ST - [0.5576, 0.4424, 0.2437])))
rows.append(("Sobol indices, Ishigami function (N = 4096)", "max abs. error", err, 3e-2))

print("| Check | Metric | Result | Tolerance |\n| --- | --- | --- | --- |")
for name, metric, val, tol in rows:
    flag = "pass" if val <= tol else "FAIL"
    print(f"| {name} | {metric} | {val:.1e} ({flag}) | {tol:.0e} |")
