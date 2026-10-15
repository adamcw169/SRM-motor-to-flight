"""Calibrate the Cesaroni L1395 BATES model to its certified thrust curve.

The trace fixes c*; the pressure exponent n and throat erosion k trade off against each other,
so (a_ref, c*, k) are fitted at each n across Davenas' range for non-aluminised composites.

Run:  python validation/ndrt/fetch_data.py && python examples/03_ndrt_calibration.py
      -> validation/ndrt/calibration.json, figures/ndrt_calibration.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

import _style
from srmflight.calibration import read_eng
from srmflight.cases import ndrt2020 as nd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/ndrt/data"
_style.apply()

curve = read_eng(DATA / "Cesaroni_4895L1395-P.eng")
cal = nd.calibrate(curve)
cal.save(ROOT / "validation/ndrt/calibration.json")

print(f"{'n':>5} {'a_ref mm/s':>11} {'c* m/s':>8} {'k mm/s':>7} {'RMS %':>6} {'dI %':>6}")
for row in zip(cal.n, cal.a_ref, cal.c_star, cal.k_erosion, cal.rms, cal.impulse_error):
    n, a, cs, k, rms, di = row
    print(f"{n:5.3f} {1e3 * a:11.3f} {cs:8.1f} {1e3 * k:7.3f} {100 * rms:6.2f} {100 * di:6.2f}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
ax0.plot(curve.t, curve.thrust, "o", ms=6, mfc=_style.SURFACE, mec=_style.INK, mew=1.5,
         label="certified static test", zorder=5)
for n, c in [(cal.n[0], _style.SERIES[0]), (cal.n[-1], _style.SERIES[1])]:
    r = nd.l1395(**cal.at(n))
    ax0.plot(r.t, r.thrust, color=c, label=f"srmflight, n = {n:.2f}")
ax0.axvspan(0, 0.25, color=_style.GRID, alpha=0.6, lw=0)
ax0.text(0.03, 150, "ignition\n(not fitted)", fontsize=8, color=_style.INK_2)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("thrust (N)")
ax0.set_ylim(0, 2000)
ax0.legend(loc="lower center")
ax0.set_title(f"Model reproduces the certified curve to {100 * cal.rms.min():.1f}% RMS")

ax1.plot(cal.n, 100 * cal.rms, "o-", color=_style.SERIES[0], ms=5)
ax1.set_xlabel("pressure exponent n")
ax1.set_ylabel("RMS thrust error (%)")
ax1.set_ylim(0, 3)
ax1.set_title(f"c* = {cal.c_star.min():.0f}–{cal.c_star.max():.0f} m/s at every n")
fig.tight_layout()
print("saved", _style.save(fig, "ndrt_calibration.png"))
