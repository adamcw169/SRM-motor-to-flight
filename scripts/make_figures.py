"""Generate every README figure, the animation and the key-numbers table into docs/assets/.

    python validation/ndrt/fetch_data.py      # once: NDRT data from the RocketPy repository
    python scripts/make_figures.py            # everything (about a minute)
    python scripts/make_figures.py hero gif   # selected figures only

Sources
  motor, burnback                 the motor model, run here from validation/ndrt/calibration.json
  hero, flight, temperature, gif  the motor model flown in RocketPy (needs the [flight] extra)
  sobol                           committed validation/ndrt/sobol_results.json (5,120 flights)

The temperature sweep uses 7 flights (examples/06 uses 15). Each static figure is written as
PNG (200 dpi) and SVG in a light and a dark variant (<name>-light.*, <name>-dark.*).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from srmflight.calibration import read_eng  # noqa: E402
from srmflight.cases import ndrt2020 as nd  # noqa: E402
from srmflight.geometry import (  # noqa: E402
    CircularPort, DistanceFieldSection, circle_polygon, star_polygon)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation" / "ndrt" / "data"
OUT = ROOT / "docs" / "assets"
STYLE = Path(__file__).with_name("figures.mplstyle")

# Okabe-Ito colour-blind-safe palette; blue is lightened on the dark background.
PALETTE = {
    "light": dict(blue="#0072B2", orange="#D55E00", green="#009E73", yellow="#E69F00", purple="#CC79A7"),
    "dark": dict(blue="#56B4E9", orange="#D55E00", green="#009E73", yellow="#E69F00", purple="#CC79A7"),
}
DARK_RC = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
    "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3", "axes.edgecolor": "#6e7681",
    "xtick.color": "#9198a1", "ytick.color": "#9198a1", "grid.color": "#21262d", "hatch.color": "#e6edf3",
}
THEMES = {"light": {}, "dark": DARK_RC}
N_MID = 0.40
SOBOL_LABELS = {
    "T_grain": "Grain temperature", "rate_factor": "Burn-rate deviation", "drag_factor": "Drag coefficient",
    "wind_factor": "Wind", "c_star_eff": r"$c^*$ efficiency", "pi_K": r"Temperature sensitivity $\pi_K$",
    "n": "Pressure exponent $n$", "rail_tilt": "Rail tilt",
}


def ink():
    return plt.rcParams["text.color"]


def quiet():
    return plt.rcParams["xtick.color"]


def bg():
    return plt.rcParams["axes.facecolor"]


def save(fig, name, theme, svg=True):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png")
    if svg:
        fig.savefig(OUT / f"{name}-{theme}.svg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data: run the motor model and the flights once
# ---------------------------------------------------------------------------

def series(func):
    a = np.asarray(func[:, :])
    return a[:, 0], a[:, 1]


def compute(need_flight=True, n_temps=7):
    cal = nd.Calibration.load(ROOT / "validation" / "ndrt" / "calibration.json")
    d = {"cal": cal, "curve": read_eng(DATA / "Cesaroni_4895L1395-P.eng"),
         "motors": {n: nd.l1395(**cal.at(n)) for n in (float(cal.n[0]), N_MID, float(cal.n[-1]))}}
    if not need_flight:
        return d
    env = nd.environment(DATA)
    res = d["motors"][N_MID]
    fl = nd.fly(nd.motor_from_result(res), env)
    t, z = series(fl.z)
    d["flight"] = {"t": t, "z": z - nd.ELEVATION, "vz": np.interp(t, *series(fl.vz)),
                   "az": np.interp(t, *series(fl.az)), "obj": fl}
    d["measured"] = nd.measured_flight(DATA)
    temps = np.linspace(-45.0, 60.0, n_temps)
    d["temps"] = temps
    d["temp_apogee"] = np.array([
        nd.fly(nd.motor_from_result(nd.l1395(**cal.at(N_MID), T_grain=273.15 + Tc)), env).apogee
        - nd.ELEVATION for Tc in temps])
    d["temp_motors"] = {Tc: nd.l1395(**cal.at(N_MID), T_grain=273.15 + Tc) for Tc in (-45, 21, 60)}

    t_m, z_m = d["measured"]
    s = res.summary()
    apo, apo_m = float(fl.apogee - nd.ELEVATION), float(z_m.max())
    d["numbers"] = {
        "pressure_exponent_n": N_MID,
        "peak_chamber_pressure_MPa": s["peak_pressure_MPa"],
        "peak_thrust_N": float(res.thrust.max()),
        "total_impulse_Ns": s["total_impulse_Ns"],
        "action_time_s": s["action_time_s"],
        "effective_exhaust_velocity_mps": s["delivered_isp_mps"],
        "mass_closure": s["mass_closure"],
        "throat_diameter_growth_mm": float(2e3 * (res.throat_radius[-1] - res.throat_radius[0])),
        "apogee_AGL_m": apo,
        "measured_apogee_AGL_m": apo_m,
        "apogee_error_pct": 100 * (apo - apo_m) / apo_m,
        "apogee_time_s": float(fl.apogee_time),
        "max_speed_mps": float(fl.max_speed),
        "max_mach": float(fl.max_mach_number),
        "max_acceleration_mps2": float(fl.max_acceleration),
        "rail_exit_speed_mps": float(fl.out_of_rail_velocity),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key_numbers.json").write_text(json.dumps(d["numbers"], indent=2))
    return d


# ---------------------------------------------------------------------------
# BATES grain drawing (longitudinal section, five segments)
# ---------------------------------------------------------------------------

def bates_rects(w):
    """(x0, length, r_inner, r_outer) in mm for each segment at burnt web w [m]; ends and core burn."""
    R, r0, L, gap = nd.GRAIN_OD / 2, nd.GRAIN_CORE / 2, nd.GRAIN_LENGTH, nd.GRAIN_GAP
    out = []
    for i in range(nd.N_GRAINS):
        x0 = i * (L + gap) + w
        length, ri = L - 2 * w, r0 + w
        if length > 0 and ri < R:
            out.append((1e3 * x0, 1e3 * length, 1e3 * ri, 1e3 * R))
    return out


def draw_case(ax):
    R = 1e3 * nd.GRAIN_OD / 2
    total = 1e3 * (nd.N_GRAINS * nd.GRAIN_LENGTH + (nd.N_GRAINS - 1) * nd.GRAIN_GAP)
    for s in (1, -1):
        ax.plot([0, total], [s * R, s * R], color=quiet(), lw=1.5)
    ax.plot([-8, total + 8], [0, 0], color=quiet(), lw=0.6, ls=(0, (8, 3, 2, 3)))
    ax.set_xlim(-10, total + 10)
    ax.set_ylim(-R - 6, R + 6)
    ax.grid(False)
    ax.set_xlabel("Axial position (mm)")
    ax.set_ylabel("Radius (mm, scale exaggerated)")


def draw_fronts(ax, colour, n_steps=6):
    w_max = float(nd.GRAIN.web_max)
    for k, w in enumerate(np.linspace(0, w_max, n_steps + 1)[:-1]):
        for x0, length, ri, ro in bates_rects(w):
            for s in (1, -1):
                ax.plot([x0, x0, x0 + length, x0 + length], [s * ro, s * ri, s * ri, s * ro], color=colour,
                        lw=2.0 if k == 0 else 1.0, alpha=1.0 if k == 0 else 0.6)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_thrust(ax, d, C, legend=True):
    lo, mid, hi = (d["motors"][k] for k in sorted(d["motors"]))
    tt = np.linspace(0, max(lo.t[-1], hi.t[-1]), 500)
    fa, fb = np.interp(tt, lo.t, lo.thrust, right=0), np.interp(tt, hi.t, hi.thrust, right=0)
    n_lo, n_hi = sorted(d["motors"])[0], sorted(d["motors"])[-1]
    ax.fill_between(tt, np.minimum(fa, fb), np.maximum(fa, fb), color=C["blue"], alpha=0.25, lw=0,
                    label=f"Model family, n = {n_lo:.2f}–{n_hi:.2f}")
    ax.plot(mid.t, mid.thrust, color=C["blue"], label=f"Model, n = {N_MID:.2f}")
    ax.plot(d["curve"].t, d["curve"].thrust, "o", ms=6, mfc=bg(), mec=C["orange"], mew=1.6,
            label="Certified static test", zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust (N)")
    ax.set_ylim(0, 2000)
    if legend:
        ax.legend(loc="lower center")


def plot_altitude(ax, d, C, legend=True):
    t_m, z_m = d["measured"]
    f = d["flight"]
    ax.plot(t_m, z_m, color=C["orange"], lw=1.6, label="Measured (Raven altimeter)")
    ax.plot(f["t"], f["z"], color=C["blue"], label=f"Model motor in RocketPy, n = {N_MID:.2f}")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1500)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude above ground (m)")
    if legend:
        ax.legend(loc="lower right")


def fig_hero(theme, C, d):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    draw_case(a)
    draw_fronts(a, C["blue"])
    a.set_title("1  Grain burnback (five BATES segments)")
    plot_thrust(b, d, C)
    rms = 100 * float(d["cal"].rms.min())
    b.set_title(f"2  Thrust, {rms:.1f}% RMS to certificate")
    plot_altitude(c, d, C)
    c.set_title(f"3  Flight, apogee error {d['numbers']['apogee_error_pct']:+.2f}%")
    save(fig, "hero_motor_to_flight", theme, svg=False)


def fig_motor(theme, C, d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for (n, r), col in zip(sorted(d["motors"].items()), (C["green"], C["blue"], C["purple"])):
        a.plot(r.t, r.pressure / 1e6, color=col, label=f"n = {n:.2f}")
    a.set_xlabel("Time (s)")
    a.set_ylabel("Chamber pressure (MPa)")
    a.set_ylim(bottom=0)
    a.legend(title="Pressure exponent", loc="lower center")
    a.set_title("Chamber pressure (model)")
    plot_thrust(b, d, C)
    b.axvspan(0, 0.25, color=quiet(), alpha=0.15, lw=0)
    b.text(0.03, 1900, "ignition\n(not fitted)", fontsize=8, color=quiet(), va="top")
    b.set_title("Thrust against the certified curve")
    save(fig, "pressure_thrust", theme)


def fig_burnback(theme, C, d):
    OD = 0.10
    star = DistanceFieldSection(OD, star_polygon(n_points=6, r_tip=0.030, r_valley=0.015))
    tube_num = DistanceFieldSection(OD, circle_polygon(0.02))
    tube_ref = CircularPort(OD, 0.04)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
    th = np.linspace(0, 2 * np.pi, 400)
    a.plot(0.5e3 * OD * np.cos(th), 0.5e3 * OD * np.sin(th), color=quiet(), lw=1.5)
    webs = np.linspace(0, star.web_max, 8)[:-1]
    for i, w in enumerate(webs):
        for line in star.contours(w):
            inside = np.hypot(line[:, 0], line[:, 1]) <= 0.5 * OD
            seg = np.where(inside[:, None], line, np.nan)
            a.plot(1e3 * seg[:, 0], 1e3 * seg[:, 1], color=C["blue"], lw=2.0 if i == 0 else 1.0,
                   alpha=1.0 if i == 0 else 0.6)
    a.set_aspect("equal")
    a.set_xlim(-55, 55)
    a.set_ylim(-55, 55)
    a.grid(False)
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_title(f"Six-point star, front every {1e3 * webs[1]:.1f} mm")

    w = np.linspace(0, 0.0295, 200)
    b.plot(1e3 * w, 1e3 * tube_ref.perimeter(w), color=C["orange"], label="Tube, exact")
    wt = np.linspace(0, tube_num.web_max * 0.98, 18)
    b.plot(1e3 * wt, 1e3 * tube_num.perimeter(wt), "o", ms=5, mfc=bg(), mec=C["orange"], mew=1.5,
           label="Tube, distance field")
    ws = np.linspace(0, star.web_max * 0.999, 300)
    b.plot(1e3 * ws, 1e3 * star.perimeter(ws), color=C["blue"], label="Star, distance field")
    b.set_xlabel("Burnt web (mm)")
    b.set_ylabel("Burning perimeter (mm)")
    b.set_ylim(bottom=0)
    b.legend(loc="lower left")
    err = np.max(np.abs(tube_num.perimeter(wt) - tube_ref.perimeter(wt)) / tube_ref.perimeter(wt))
    b.set_title(f"Perimeter; tube error {100 * err:.3f}%")

    wb = np.linspace(0, float(nd.GRAIN.web_max) * 0.999, 200)
    c.plot(1e3 * wb, 1e4 * np.asarray(nd.GRAIN.burn_area(wb)), color=C["green"])
    c.set_xlabel("Burnt web (mm)")
    c.set_ylabel("Burning area (cm$^2$)")
    c.set_ylim(bottom=0)
    c.set_title("L1395: five BATES segments")
    save(fig, "grain_burnback", theme)


def fig_flight(theme, C, d):
    f = d["flight"]
    t_b = float(d["motors"][N_MID].t[-1])
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.2), sharex=True, constrained_layout=True)
    plot_altitude(axes[0], d, C)
    axes[0].set_xlabel("")
    axes[1].plot(f["t"], f["vz"], color=C["blue"])
    axes[1].set_ylabel("Vertical velocity (m s$^{-1}$)")
    axes[2].plot(f["t"], f["az"], color=C["blue"])
    axes[2].axhline(0, color=quiet(), lw=0.8)
    axes[2].set_ylabel("Vertical acceleration (m s$^{-2}$)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(t_b, color=quiet(), lw=1, ls=(0, (4, 3)))
    axes[1].text(t_b + 0.2, 0.05, "burnout", fontsize=8.5, color=quiet(),
                 transform=axes[1].get_xaxis_transform())
    save(fig, "flight_profile", theme)


def fig_sobol(theme, C, d):
    s = json.loads((ROOT / "validation" / "ndrt" / "sobol_results.json").read_text())["apogee"]
    names = sorted(s["ST"], key=lambda k: s["ST"][k])
    y = np.arange(len(names))
    ST = np.array([s["ST"][k] for k in names])
    S1 = np.clip([s["S1"][k] for k in names], 0, None)
    ci = np.array([s["ST_ci"][k] for k in names])
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.barh(y + 0.19, ST, height=0.34, color=C["blue"], label="Total-order $S_T$")
    ax.errorbar(ST, y + 0.19, xerr=[np.clip(ST - ci[:, 0], 0, None), np.clip(ci[:, 1] - ST, 0, None)],
                fmt="none", ecolor=ink(), lw=0.9, capsize=2)
    ax.barh(y - 0.19, S1, height=0.34, color=C["yellow"], label="First-order $S_1$")
    ax.set_yticks(y)
    ax.set_yticklabels([SOBOL_LABELS[k] for k in names])
    ax.set_xlabel(f"Share of apogee variance (–); apogee standard deviation {s['std']:.1f} m")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "sobol_apogee", theme)


def fig_temperature(theme, C, d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for (Tc, r), col in zip(sorted(d["temp_motors"].items()), (C["blue"], C["green"], C["orange"])):
        a.plot(r.t, r.thrust, color=col, label=f"{Tc:+d} °C")
    a.set_xlabel("Time (s)")
    a.set_ylabel("Thrust (N)")
    a.set_ylim(0, 2000)
    a.legend(loc="lower left", title="Grain temperature")
    b.plot(d["temps"], d["temp_apogee"], "o-", color=C["blue"], mec=bg())
    b.set_xlabel("Grain temperature (°C)")
    b.set_ylabel("Apogee above ground (m)")
    save(fig, "temperature_sweep", theme)


# ---------------------------------------------------------------------------
# Animation: grain burnback and flight against time
# ---------------------------------------------------------------------------

def gif_flight(theme, C, d, dt=0.2):
    res, f = d["motors"][N_MID], d["flight"]
    t_end = float(f["t"][-1])
    times = np.arange(0.0, t_end + dt, dt)
    fig = plt.figure(figsize=(9.6, 5.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], left=0.08, right=0.98, top=0.88, bottom=0.11,
                          hspace=0.18, wspace=0.24)
    g = fig.add_subplot(gs[:, 0])
    draw_case(g)
    g.set_title("L1395 grain, longitudinal section")
    patches = []
    for _ in range(nd.N_GRAINS):
        for _s in (1, -1):
            p = Rectangle((0, 0), 0, 0, facecolor=C["blue"], edgecolor="none", alpha=0.85)
            g.add_patch(p)
            patches.append(p)
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    specs = [(res.t, res.thrust, "Thrust (N)", (0, 2000), C["orange"]),
             (f["t"], f["z"], "Altitude (m)", (0, 1500), C["blue"]),
             (f["t"], f["vz"], "Velocity (m s$^{-1}$)", (0, 200), C["green"])]
    lines, dots = [], []
    for ax, (tx, yx, lab, lim, col) in zip(axes, specs):
        ax.plot(tx, yx, color=col, lw=1, alpha=0.25)
        ln, = ax.plot([], [], color=col)
        dot, = ax.plot([], [], "o", color=col, ms=5)
        lines.append(ln)
        dots.append(dot)
        ax.set_xlim(0, t_end)
        ax.set_ylim(*lim)
        ax.set_ylabel(lab, fontsize=9)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time (s)")
    title = fig.text(0.08, 0.94, "", fontsize=11, fontweight="bold")
    frames = list(range(len(times))) + [len(times) - 1] * 12

    def draw(i):
        t = times[i]
        w = float(np.interp(t, res.t, res.web, right=res.web[-1])) if t < res.t[-1] else 1.0
        rects = bates_rects(w)
        for k in range(nd.N_GRAINS):
            for j, s in enumerate((1, -1)):
                p = patches[2 * k + j]
                if k < len(rects):
                    x0, length, ri, ro = rects[k]
                    p.set_bounds(x0, s * ri if s > 0 else -ro, length, ro - ri)
                    p.set_visible(True)
                else:
                    p.set_visible(False)
        for ln, dot, (tx, yx, *_rest) in zip(lines, dots, specs):
            m = tx <= t
            ln.set_data(tx[m], yx[m])
            dot.set_data([t], [np.interp(t, tx, yx, right=0.0)])
        phase = "burning" if t < res.t[-1] else "coasting"
        title.set_text(f"NDRT 2020, model motor in RocketPy   t = {t:4.1f} s   {phase}")
        return patches + lines + dots + [title]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / f"flight-{theme}.gif", writer=PillowWriter(fps=12), dpi=120)
    plt.close(fig)


FIGS = {"hero": fig_hero, "motor": fig_motor, "burnback": fig_burnback, "flight": fig_flight,
        "sobol": fig_sobol, "temperature": fig_temperature, "gif": gif_flight}
NEEDS_FLIGHT = {"hero", "flight", "temperature", "gif"}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(FIGS)
    data = compute(need_flight=bool(NEEDS_FLIGHT & set(todo)))
    for theme, rc in THEMES.items():
        with plt.style.context([str(STYLE), rc]):
            for k in todo:
                FIGS[k](theme, PALETTE[theme], data)
                print(f"wrote {k} ({theme})")
    if "numbers" in data:
        print(json.dumps(data["numbers"], indent=2))
        print("temperature sweep (C, apogee m):",
              [(round(float(a), 1), round(float(b), 1)) for a, b in zip(data["temps"], data["temp_apogee"])])
        print("action time by temperature (s):",
              {k: round(v.action_time(), 2) for k, v in data["temp_motors"].items()})
