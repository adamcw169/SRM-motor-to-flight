"""Grain temperature across the -45 to +60 C envelope: motor thrust and NDRT apogee.

Run:  python examples/06_ndrt_temperature.py   ->  figures/ndrt_temperature.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from srmflight.cases import ndrt2020 as nd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/ndrt/data"
_style.apply()

cal = nd.Calibration.load(ROOT / "validation/ndrt/calibration.json")
params = cal.at(0.40)
env = nd.environment(DATA)

temps_c = np.linspace(-45, 60, 15)
rows = []
for Tc in temps_c:
    res = nd.l1395(**params, T_grain=273.15 + Tc)
    fl = nd.fly(nd.motor_from_result(res), env)
    rows.append((Tc, res.peak_pressure / 1e6, res.action_time(), res.total_impulse,
                 fl.apogee - nd.ELEVATION))
rows = np.array(rows)
for r in rows:
    print(f"T {r[0]:6.1f} C  p_max {r[1]:5.2f} MPa  t_a {r[2]:5.2f} s  I {r[3]:6.0f} N s"
          f"  apogee {r[4]:7.1f} m")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))
for Tc, c in [(-45, _style.SERIES[0]), (21, _style.INK_2), (60, _style.SERIES[1])]:
    res = nd.l1395(**params, T_grain=273.15 + Tc)
    ax0.plot(res.t, res.thrust, color=c, label=f"{Tc:+d} °C")
ax0.set_xlabel("time (s)")
ax0.set_ylabel("thrust (N)")
ax0.set_ylim(0, 2000)
ax0.legend(loc="lower left", title="grain temperature")
span = rows[:, 2].max() - rows[:, 2].min()
ax0.set_title(f"Burn time shifts by {span:.1f} s across the envelope")

ax1.plot(rows[:, 0], rows[:, 4], "o-", color=_style.SERIES[0], ms=4)
ax1.set_xlabel("grain temperature (°C)")
ax1.set_ylabel("apogee above ground (m)")
dz = rows[-1, 4] - rows[0, 4]
ax1.set_title(f"Apogee rises {dz:.0f} m from −45 to +60 °C")
fig.tight_layout()
print("saved", _style.save(fig, "ndrt_temperature.png"))
