"""Which uncertainties move the NDRT flight? Sobol indices of apogee and launch-phase outputs.

Every sample runs the calibrated motor model, then a RocketPy 6-DOF flight to apogee.

Motor inputs
    n            pressure exponent, U(0.30, 0.50); a_ref, c*, k follow the calibration
                 (Davenas 1993 ch. 2 Table 3, non-aluminised composite range)
    T_grain      grain temperature, U(-45, +60) C (Davenas ch. 1 §2.6, tactical envelope)
    pi_K         temperature sensitivity, U(0.15, 0.35) %/C around Davenas' typical 0.25
    rate_factor  burn-rate deviation r/r0, U(0.95, 1.05) (Davenas ch. 4 §3.1.2, hump effect)
    c_star_eff   delivered-c* scatter, N(1, 0.0037): sigma(Isp) 0.5-1 s (Davenas ch. 3)
Airframe and launch inputs (RocketPy NDRT 2020 acceptance case)
    drag_factor  N(1, 0.033)      wind_factor  N(1, 0.30)      rail_tilt  |N(0, 1 deg)|

Run:  python examples/05_ndrt_flight_sobol.py   (about 20 min for N = 512)
      -> figures/ndrt_sobol.png, validation/ndrt/sobol_results.json
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import _style
from srmflight.cases import ndrt2020 as nd
from srmflight.uq import run_sobol

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/ndrt/data"
N_BASE = 512
_style.apply()

cal = nd.Calibration.load(ROOT / "validation/ndrt/calibration.json")
DISTS = {
    "n": stats.uniform(0.30, 0.20),
    "T_grain": stats.uniform(228.15, 105.0),
    "pi_K": stats.uniform(0.0015, 0.0020),
    "rate_factor": stats.uniform(0.95, 0.10),
    "c_star_eff": stats.norm(1.0, 0.0037),
    "drag_factor": stats.norm(1.0, 0.033),
    "wind_factor": stats.norm(1.0, 0.30),
    "rail_tilt": stats.halfnorm(scale=1.0),
}
LABELS = {"n": "pressure exponent n", "T_grain": "grain temperature", "pi_K": "temp. sensitivity",
          "rate_factor": "burn-rate deviation", "c_star_eff": "c* scatter",
          "drag_factor": "drag coefficient", "wind_factor": "wind speed", "rail_tilt": "rail tilt"}
MOTOR = ("n", "T_grain", "pi_K", "rate_factor", "c_star_eff")


def flight(p: dict) -> dict:
    res = nd.l1395(**cal.at(p["n"]), T_grain=p["T_grain"], pi_k=p["pi_K"],
                   rate_factor=p["rate_factor"], c_star_eff=p["c_star_eff"], n_out=500)
    env = nd.environment(DATA, wind_factor=max(p["wind_factor"], 0.0))
    fl = nd.fly(nd.motor_from_result(res), env, drag_factor=p["drag_factor"],
                inclination=90.0 - p["rail_tilt"])
    return {"apogee": fl.apogee - nd.ELEVATION, "max speed": fl.max_speed,
            "rail exit speed": fl.out_of_rail_velocity, "apogee time": fl.apogee_time}


t0 = time.time()
results = run_sobol(flight, DISTS, n_base=N_BASE, seed=11)
runs = N_BASE * (len(DISTS) + 2)
print(f"{runs} motor + flight runs in {(time.time() - t0) / 60:.1f} min")

out = {}
for name, r in results.items():
    print(f"\n{name}  (std {np.sqrt(r.variance):.3g})\n{r.table()}")
    out[name] = {"std": float(np.sqrt(r.variance)), "S1": dict(zip(r.names, map(float, r.S1))),
                 "ST": dict(zip(r.names, map(float, r.ST))),
                 "ST_ci": {k: list(map(float, v)) for k, v in zip(r.names, r.ST_ci)}}
out["_meta"] = {"n_base": N_BASE, "runs": runs}
(ROOT / "validation/ndrt/sobol_results.json").write_text(json.dumps(out, indent=2))

shown = ["apogee", "max speed", "rail exit speed"]
units = {"apogee": "m", "max speed": "m/s", "rail exit speed": "m/s"}
order = sorted(DISTS, key=lambda k: -results["apogee"].ST[list(DISTS).index(k)])
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
y = np.arange(len(order))
for ax, key in zip(axes, shown):
    r = results[key]
    idx = [r.names.index(k) for k in order]
    st, s1, ci = r.ST[idx], np.clip(r.S1[idx], 0, None), r.ST_ci[idx]
    colors = [_style.SERIES[1] if k in MOTOR else _style.SERIES[0] for k in order]
    ax.barh(y, st, height=0.62, color=colors)
    ax.errorbar(st, y, xerr=np.abs(ci.T - st), fmt="none", ecolor=_style.INK_2, lw=1, capsize=2)
    ax.plot(s1, y, "|", color=_style.INK, ms=12, mew=1.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("total-order Sobol index $S_T$")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{key}  (σ = {np.sqrt(r.variance):.1f} {units[key]})")
axes[0].set_yticks(y, [LABELS[k] for k in order])
axes[0].invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=_style.SERIES[1]),
           plt.Rectangle((0, 0), 1, 1, color=_style.SERIES[0]),
           plt.Line2D([], [], color=_style.INK, marker="|", ls="", ms=12, mew=1.6)]
axes[-1].legend(handles, ["motor input", "airframe / launch input", "first order $S_1$"],
                loc="lower right")
top = LABELS[order[0]]
fig.suptitle(f"{top[0].upper() + top[1:]} dominates apogee scatter "
             f"(σ = {np.sqrt(results['apogee'].variance):.0f} m over {runs} flights)",
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=_style.INK)
fig.tight_layout()
print("saved", _style.save(fig, "ndrt_sobol.png"))
