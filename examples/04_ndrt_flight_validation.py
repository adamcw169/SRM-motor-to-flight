"""Fly NDRT 2020 with the calibrated motor model and compare with the measured flight.

Run:  python examples/04_ndrt_flight_validation.py
      -> figures/ndrt_flight.png, validation/ndrt/flight_validation.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style
from srmflight.cases import ndrt2020 as nd
from srmflight.flight import exhaust_velocity_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/ndrt/data"
_style.apply()

cal = nd.Calibration.load(ROOT / "validation/ndrt/calibration.json")
env = nd.environment(DATA)
t_meas, z_meas = nd.measured_flight(DATA)
apo_meas, t_apo_meas = float(z_meas.max()), float(t_meas[np.argmax(z_meas)])

certified = nd.fly(nd.rocketpy_motor(str(DATA / "Cesaroni_4895L1395-P.eng"), 3.45), env)
runs = {}
for n in (cal.n[0], 0.40, cal.n[-1]):
    res = nd.l1395(**cal.at(n))
    runs[n] = (res, nd.fly(nd.motor_from_result(res), env))


def apogee(fl):
    return fl.apogee - nd.ELEVATION


rows = {"measured": {"apogee_m": apo_meas, "apogee_time_s": t_apo_meas}}
rows["RocketPy, certified curve"] = {"apogee_m": apogee(certified),
                                     "apogee_time_s": certified.apogee_time}
for n, (res, fl) in runs.items():
    rows[f"srmflight motor, n = {n:.2f}"] = {
        "apogee_m": apogee(fl), "apogee_time_s": fl.apogee_time,
        "max_speed_mps": fl.max_speed, "constant_ve_mass_error": exhaust_velocity_error(res)}
for k, v in rows.items():
    v["apogee_error_pct"] = 100 * (v["apogee_m"] - apo_meas) / apo_meas
    print(f"{k:<30} apogee {v['apogee_m']:7.1f} m ({v['apogee_error_pct']:+.2f}%)  "
          f"t_apo {v['apogee_time_s']:5.2f} s")
(ROOT / "validation/ndrt/flight_validation.json").write_text(json.dumps(rows, indent=2))

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(t_meas, z_meas, color=_style.INK, lw=1.6, label="measured (Raven altimeter)")
zc = certified.z[:, :]
ax.plot(zc[:, 0], zc[:, 1] - nd.ELEVATION, "--", color=_style.INK_2, lw=1.5,
        label="RocketPy, certified thrust curve")
lo, hi = runs[cal.n[0]][1].z[:, :], runs[cal.n[-1]][1].z[:, :]
tt = np.linspace(0, max(lo[-1, 0], hi[-1, 0]), 400)
za, zb = (np.interp(tt, z[:, 0], z[:, 1] - nd.ELEVATION) for z in (lo, hi))
ax.fill_between(tt, np.minimum(za, zb), np.maximum(za, zb), color=_style.SERIES[0], alpha=0.25,
                lw=0, label=f"srmflight motor, n = {cal.n[0]:.2f}–{cal.n[-1]:.2f}")
mid = runs[0.40][1].z[:, :]
ax.plot(mid[:, 0], mid[:, 1] - nd.ELEVATION, color=_style.SERIES[0], lw=2,
        label="srmflight motor, n = 0.40")
ax.set_xlim(0, 20)
ax.set_ylim(0, 1450)
ax.set_xlabel("time (s)")
ax.set_ylabel("altitude above ground (m)")
ax.legend(loc="upper left")
errs = [v["apogee_error_pct"] for k, v in rows.items() if k.startswith("srmflight")]
ax.set_title(f"Calibrated motor model lands within {max(map(abs, errs)):.1f}% of measured apogee")

ins = ax.inset_axes([0.42, 0.12, 0.28, 0.42])
ins.plot(t_meas, z_meas, color=_style.INK, lw=1.4)
ins.plot(zc[:, 0], zc[:, 1] - nd.ELEVATION, "--", color=_style.INK_2, lw=1.3)
ins.fill_between(tt, np.minimum(za, zb), np.maximum(za, zb), color=_style.SERIES[0], alpha=0.25,
                 lw=0)
ins.plot(mid[:, 0], mid[:, 1] - nd.ELEVATION, color=_style.SERIES[0], lw=1.8)
ins.set_xlim(14.5, 18)
ins.set_ylim(1285, 1335)
ins.tick_params(labelsize=7)
ins.set_title("apogee", fontsize=8, loc="left")
fig.tight_layout()
print("saved", _style.save(fig, "ndrt_flight.png"))
