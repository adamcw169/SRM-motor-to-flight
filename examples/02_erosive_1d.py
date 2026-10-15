"""Erosive burning in a slender tube motor: 0D vs quasi-1D with two erosive models.

Run:  python examples/02_erosive_1d.py   ->  figures/erosive_1d.png
"""

import matplotlib.pyplot as plt

import _style
from srmflight import LenoirRobillard, MukundaPaul, Nozzle, NoErosion, Propellant
from srmflight.ballistics import QuasiOneDMotor, ZeroDMotor
from srmflight.geometry import AxialGrain, CircularPort, Grain

_style.apply()
prop = Propellant(density=1750, a_ref=0.007, n=0.35, c_star=1500, gamma=1.2)
section = CircularPort(outer_diameter=0.10, port_diameter=0.05)
nozzle = Nozzle(throat_diameter=0.036, exit_diameter=0.075)
L = 1.6  # L/D_port = 32, port-to-throat area ratio 1.9
lr = LenoirRobillard.from_properties(cp_gas=2000, mu_gas=8e-5, pr_gas=0.5, rho_p=1750,
                                     cp_solid=1250, T_flame=3000, T_surface=1000,
                                     T_grain=294.15)

runs = {"0D (no axial flow)": ZeroDMotor(Grain.single(section, L), prop, nozzle).run()}
grain = AxialGrain.uniform(section, L, n_stations=100)
for label, model in [("quasi-1D, no erosion", NoErosion()),
                     ("quasi-1D, Lenoir–Robillard", lr),
                     ("quasi-1D, Mukunda–Paul", MukundaPaul())]:
    runs[label] = QuasiOneDMotor(grain, prop, nozzle, erosive=model).run(dw_fraction=1 / 300)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.2))
colors = [_style.INK_2, _style.SERIES[0], _style.SERIES[1], _style.SERIES[2]]
for (label, res), c in zip(runs.items(), colors):
    ls = "--" if label.startswith("0D") else "-"
    ax0.plot(res.t, res.pressure / 1e6, ls, color=c, label=label, lw=1.6 if ls == "--" else 2)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("head-end pressure (MPa)")
ax0.set_ylim(bottom=0)
ax0.legend(loc="lower center")
ax0.set_title("Erosion raises start pressure, burns aft end out early")

for (label, res), c in zip(list(runs.items())[1:], colors[1:]):
    t0, p, r, _ = res.extra["profiles"][0]
    ax1.plot(1e2 * grain.z, 1e3 * r, color=c, label=label)
ax1.set_xlabel("distance from head end (cm)")
ax1.set_ylabel("burn rate at ignition (mm/s)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left")
ax1.set_title("Burn rate along the grain at t = 0")
fig.tight_layout()
print("saved", _style.save(fig, "erosive_1d.png"))
for label, res in runs.items():
    s = res.summary()
    print(f"{label:<28} p_max {s['peak_pressure_MPa']:6.2f} MPa  I {s['total_impulse_Ns']:8.0f} N s"
          f"  t_a {s['action_time_s']:5.2f} s")
