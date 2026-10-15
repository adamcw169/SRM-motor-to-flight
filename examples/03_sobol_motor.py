"""Which uncertainties drive motor performance? Sobol indices for a star-grain motor.

Inputs (illustrative ranges, to be replaced with literature-backed distributions):
    a_ref      burn-rate coefficient, +/-3 % (1 sigma) batch scatter
    n          pressure exponent, 0.35 +/- 0.01
    T_grain    grain temperature, uniform -40 C to +60 C (typical tactical envelope)
    k_throat   throat recession coefficient, uniform 0-0.2 mm/s at 10 MPa
    c_eff      c* efficiency, 0.97 +/- 0.01

Run:  python examples/03_sobol_motor.py   ->  figures/sobol_motor.png
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import _style
from srmflight import Nozzle, Propellant, ThroatErosion
from srmflight.ballistics import ZeroDMotor
from srmflight.geometry import DistanceFieldSection, Grain, star_polygon
from srmflight.uq import run_sobol

_style.apply()
BASE = Propellant(density=1750, a_ref=0.007, n=0.35, c_star=1500, gamma=1.2, sigma_p=0.002)
GRAIN = Grain.single(DistanceFieldSection(0.10, star_polygon(6, 0.030, 0.015)), length=0.6)

DISTS = {
    "a_ref": stats.norm(0.007, 0.007 * 0.03),
    "n": stats.norm(0.35, 0.01),
    "T_grain": stats.uniform(233.15, 100.0),
    "k_throat": stats.uniform(0.0, 2.0e-5),
    "c_eff": stats.norm(0.97, 0.01),
}


def motor(p: dict) -> dict:
    prop = BASE.with_(a_ref=p["a_ref"], n=p["n"], c_star_efficiency=p["c_eff"])
    noz = Nozzle(0.024, 0.06, erosion=ThroatErosion(k=p["k_throat"], m=1.0, p_ref=1e6))
    res = ZeroDMotor(GRAIN, prop, noz, T_grain=p["T_grain"]).run(n_out=600)
    return {"peak pressure": res.peak_pressure, "total impulse": res.total_impulse,
            "action time": res.action_time()}


t0 = time.time()
N_BASE = 1024
results = run_sobol(motor, DISTS, n_base=N_BASE, seed=7)
print(f"{N_BASE * (len(DISTS) + 2)} motor runs in {time.time() - t0:.0f} s")

labels = {"a_ref": "burn-rate coeff. a", "n": "pressure exp. n", "T_grain": "grain temperature",
          "k_throat": "throat erosion k", "c_eff": "c* efficiency"}
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
y = np.arange(len(DISTS))
for ax, (out, res) in zip(axes, results.items()):
    print(f"\n{out}\n{res.table()}")
    ax.barh(y + 0.19, res.ST, height=0.36, color=_style.SERIES[1], label="total order $S_T$")
    ax.barh(y - 0.19, np.clip(res.S1, 0, None), height=0.36, color=_style.SERIES[0],
            label="first order $S_1$")
    ax.errorbar(res.ST, y + 0.19, xerr=np.abs(res.ST_ci.T - res.ST), fmt="none",
                ecolor=_style.INK_2, lw=1, capsize=2)
    ax.set_title(out)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of output variance")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, [labels[k] for k in DISTS])
axes[0].invert_yaxis()
axes[-1].legend(loc="lower right")
top = {out: labels[list(DISTS)[int(np.argmax(r.ST))]] for out, r in results.items()}
fig.suptitle(f"{top['peak pressure'].capitalize()} drives peak pressure and action time; "
             f"{top['total impulse']} drives impulse", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=_style.INK)
fig.tight_layout()
print("saved", _style.save(fig, "sobol_motor.png"))
