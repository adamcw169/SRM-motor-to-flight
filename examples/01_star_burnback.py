"""Star-grain burnback via the signed-distance field, checked against a tube grain.

Run:  python examples/01_star_burnback.py   ->  figures/burnback_star.png
"""

import matplotlib.pyplot as plt
import numpy as np

import _style
from srmflight.geometry import CircularPort, DistanceFieldSection, circle_polygon, star_polygon

_style.apply()
OD = 0.10
star = DistanceFieldSection(OD, star_polygon(n_points=6, r_tip=0.030, r_valley=0.015))
tube_num = DistanceFieldSection(OD, circle_polygon(0.02))
tube_ref = CircularPort(OD, 0.04)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})

# --- burning surface at even web steps
theta = np.linspace(0, 2 * np.pi, 400)
ax0.plot(0.5e3 * OD * np.cos(theta), 0.5e3 * OD * np.sin(theta), color=_style.INK_2, lw=1.2)
webs = np.linspace(0, star.web_max, 8)[:-1]
for i, w in enumerate(webs):
    for line in star.contours(w):
        inside = np.hypot(line[:, 0], line[:, 1]) <= 0.5 * OD
        seg = np.where(inside[:, None], line, np.nan)
        ax0.plot(1e3 * seg[:, 0], 1e3 * seg[:, 1], color=_style.SERIES[0],
                 lw=2.0 if i == 0 else 1.0, alpha=1.0 if i == 0 else 0.55)
ax0.set_aspect("equal")
ax0.set_xlim(-55, 55)
ax0.set_ylim(-55, 55)
ax0.grid(False)
ax0.set_xticks([])
ax0.set_yticks([])
for s in ax0.spines.values():
    s.set_visible(False)
ax0.set_title("Six-point star, burning surface every %.1f mm" % (1e3 * webs[1]))

# --- perimeter vs web
w = np.linspace(0, 0.0305, 200)
ax1.plot(1e3 * w, 1e3 * tube_ref.perimeter(w), color=_style.SERIES[1], lw=2, label="tube, exact")
wt = np.linspace(0, tube_num.web_max * 0.98, 18)
ax1.plot(1e3 * wt, 1e3 * tube_num.perimeter(wt), "o", ms=5, mfc=_style.SURFACE,
         mec=_style.SERIES[1], mew=1.5, label="tube, distance field")
ws = np.linspace(0, star.web_max * 0.999, 300)
ax1.plot(1e3 * ws, 1e3 * star.perimeter(ws), color=_style.SERIES[0], lw=2,
         label="star, distance field")
ax1.set_xlabel("burnt web (mm)")
ax1.set_ylabel("burning perimeter (mm)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="lower left")
err = np.max(np.abs(tube_num.perimeter(wt) - tube_ref.perimeter(wt)) / tube_ref.perimeter(wt))
ax1.set_title(f"Star is progressive, then a sliver; tube error {100 * err:.3f}%")
fig.tight_layout()
print("saved", _style.save(fig, "burnback_star.png"))
