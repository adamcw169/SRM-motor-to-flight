"""NDRT 2020 flight (Notre Dame Rocket Team, 23 Feb 2020): motor model and RocketPy setup.

Airframe, launch and recovery values are those of RocketPy's NDRT 2020 acceptance case
(Ceotto et al. 2021; RocketPy ``tests/acceptance/test_ndrt_2020_rocket.py``). The motor is a
Cesaroni L1395-BS: five BATES grains (OD 66 mm, core 30 mm, length 120 mm), 21.5 mm throat,
49.5 mm exit, 2.475 kg propellant (certificate). Its burn-rate law and c* are not published;
they are calibrated to the certified thrust curve by ``calibrate``.

RocketPy is an optional dependency (``pip install 'srmflight[flight]'``).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ballistics import ZeroDMotor
from ..calibration import EngCurve, fit_thrust_curve
from ..geometry import CircularPort, Grain
from ..nozzle import Nozzle, ThroatErosion
from ..propellant import Propellant
from ..results import MotorResult

# ----------------------------------------------------------------- motor (Cesaroni L1395)
GRAIN_OD, GRAIN_CORE, GRAIN_LENGTH, N_GRAINS, GRAIN_GAP = 0.066, 0.030, 0.120, 5, 0.003
THROAT_D, EXIT_D = 0.0215, 0.0495
PROPELLANT_MASS = 2.475  # kg, certificate
GAMMA = 1.20  # combustion-gas ratio of specific heats (assumed; enters C_F only)
NOZZLE_EFF = 0.95  # thrust-coefficient efficiency (assumed; c* absorbs the rest)
P_REF = 6.9e6
T_REF = 294.15  # K, assumed static-test grain temperature (21 C)
N_RANGE = (0.30, 0.50)  # non-aluminised composite propellants (Davenas 1993, ch. 2, Table 3)
PI_K = 0.0025  # 1/K, typical composite sensitivity at constant K_n (Davenas, ch. 1 §2.6)

GRAIN = Grain.bates(CircularPort(GRAIN_OD, GRAIN_CORE), GRAIN_LENGTH, N_GRAINS)
DENSITY = PROPELLANT_MASS / float(GRAIN.propellant_volume(0.0))


def l1395(a_ref: float, n: float, c_star: float, k_erosion: float, T_grain: float = T_REF,
          pi_k: float = PI_K, rate_factor: float = 1.0, c_star_eff: float = 1.0,
          n_out: int = 800) -> MotorResult:
    """Run the L1395 model.

    ``k_erosion`` is the throat recession rate [m/s] at 6.9 MPa (linear in pressure).
    Temperature sensitivity: sigma_p = (1 - n) * pi_K (Davenas, ch. 1 §2.6).
    """
    prop = Propellant(density=DENSITY, a_ref=a_ref * rate_factor, n=n, c_star=c_star,
                      gamma=GAMMA, p_ref=P_REF, T_ref=T_REF, sigma_p=(1.0 - n) * pi_k,
                      c_star_efficiency=c_star_eff)
    noz = Nozzle(THROAT_D, EXIT_D, efficiency=NOZZLE_EFF,
                 erosion=ThroatErosion(k=k_erosion, m=1.0, p_ref=P_REF))
    return ZeroDMotor(GRAIN, prop, noz, T_grain=T_grain).run(n_out=n_out)


@dataclass
class Calibration:
    """Calibrated (a_ref, c*, k) as a function of the pressure exponent n."""

    n: np.ndarray
    a_ref: np.ndarray
    c_star: np.ndarray
    k_erosion: np.ndarray
    rms: np.ndarray
    impulse_error: np.ndarray

    def at(self, n: float) -> dict:
        f = lambda y: float(np.interp(n, self.n, y))  # noqa: E731
        return {"a_ref": f(self.a_ref), "n": float(n), "c_star": f(self.c_star),
                "k_erosion": f(self.k_erosion)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({k: list(map(float, getattr(self, k))) for k in
                                          ("n", "a_ref", "c_star", "k_erosion", "rms",
                                           "impulse_error")}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(**{k: np.array(v) for k, v in d.items()})


def calibrate(curve: EngCurve, n_values=np.linspace(*N_RANGE, 9),
              window=(0.25, 3.0)) -> Calibration:
    """Fit (a_ref, c*, k) at each fixed n. The trace fixes c* but trades n against erosion."""
    rows = []
    for n in n_values:
        def run(p, n=n):
            return l1395(p["a_ref"], n, p["c_star"], p["k"] * 1e-3, n_out=600)

        f = fit_thrust_curve(run, curve, {"a_ref": 0.008, "c_star": 1430.0, "k": 1.0},
                             {"a_ref": (0.002, 0.03), "c_star": (900.0, 1800.0), "k": (0.0, 3.0)},
                             window=window)
        rows.append((n, f.params["a_ref"], f.params["c_star"], f.params["k"] * 1e-3,
                     f.rms_error, f.impulse_error))
    cols = np.array(rows).T
    return Calibration(*cols)


# ----------------------------------------------------------------- environment
LAT, LON, ELEVATION = 41.775447, -86.572467, 206.0
LAUNCH = dt.datetime(2020, 2, 23, 16)  # UTC
RAIL_LENGTH, INCLINATION, HEADING = 3.353, 90.0, 181.0


def era5_profile(path: str | Path, lat: float = LAT, lon: float = LON, when=LAUNCH,
                 earth_radius: float = 6378137.0):
    """Height, pressure, temperature and wind profiles from an ERA5 pressure-level file.

    Mirrors RocketPy's reanalysis reader: nearest time, bilinear interpolation in latitude
    and longitude, geopotential converted to geometric height. Reads NetCDF3 with SciPy, so
    no netCDF4/HDF5 dependency is needed.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import netcdf_file

    with netcdf_file(str(path), "r", mmap=False) as f:
        def field(name):
            v = f.variables[name]
            return v[:] * getattr(v, "scale_factor", 1.0) + getattr(v, "add_offset", 0.0)

        hours = f.variables["time"][:].astype(float)
        lats = f.variables["latitude"][:].astype(float)
        lons = f.variables["longitude"][:].astype(float)
        levels = f.variables["level"][:].astype(float) * 100.0  # hPa -> Pa
        query = (when - dt.datetime(1900, 1, 1)).total_seconds() / 3600.0
        it = int(np.argmin(np.abs(hours - query)))
        order = np.argsort(lats)

        def at_site(name):
            F = field(name)[it][:, order, :]
            return np.array([float(RegularGridInterpolator((lats[order], lons), F[k])(
                [lat, lon])[0]) for k in range(F.shape[0])])

        zgp = at_site("z") / 9.80665
        h = earth_radius * zgp / (earth_radius - zgp)
        T, u, v = at_site("t"), at_site("u"), at_site("v")
    s = np.argsort(h)
    return h[s], levels[s], T[s], u[s], v[s]


_PROFILES: dict[str, tuple] = {}


def environment(data_dir: str | Path, wind_factor: float = 1.0):
    """RocketPy Environment for the launch, with the ERA5 wind profile scaled by wind_factor."""
    from rocketpy import Environment

    key = str(Path(data_dir).resolve())
    if key not in _PROFILES:
        _PROFILES[key] = era5_profile(Path(data_dir) / "ndrt_2020_weather_data_ERA5.nc")
    h, p, T, u, v = _PROFILES[key]
    env = Environment(gravity=9.81, latitude=LAT, longitude=LON,
                      date=(LAUNCH.year, LAUNCH.month, LAUNCH.day, LAUNCH.hour),
                      elevation=ELEVATION)
    col = lambda y: np.column_stack([h, y]).tolist()  # noqa: E731
    env.set_atmospheric_model(type="custom_atmosphere", pressure=col(p), temperature=col(T),
                              wind_u=col(wind_factor * u), wind_v=col(wind_factor * v))
    env.max_expected_height = 2500
    return env


# ----------------------------------------------------------------- rocket
NOZZLE_POSITION, GRAINS_CM_POSITION = -1.255, -0.85704


def rocketpy_motor(thrust_source, burn_time: float):
    """RocketPy SolidMotor with the L1395 grain geometry and a given thrust source."""
    from rocketpy import SolidMotor

    return SolidMotor(
        thrust_source=thrust_source, burn_time=burn_time, dry_mass=1.0, dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=0, grains_center_of_mass_position=GRAINS_CM_POSITION,
        grain_number=N_GRAINS, grain_separation=GRAIN_GAP, grain_density=DENSITY,
        grain_outer_radius=GRAIN_OD / 2, grain_initial_inner_radius=GRAIN_CORE / 2,
        grain_initial_height=GRAIN_LENGTH, nozzle_radius=EXIT_D / 2, throat_radius=THROAT_D / 2,
        interpolation_method="linear", nozzle_position=NOZZLE_POSITION)


def motor_from_result(result: MotorResult):
    from ..flight import thrust_curve

    curve = thrust_curve(result)
    return rocketpy_motor(curve, float(curve[-1, 0]))


def rocket(motor, drag_factor: float = 1.0):
    from rocketpy import Rocket

    rk = Rocket(radius=0.203 / 2, mass=23.321 - 2.475 - 1.0, inertia=(83.351, 83.351, 0.15982),
                power_off_drag=0.44 * drag_factor, power_on_drag=0.44 * drag_factor,
                center_of_mass_without_motor=0)
    rk.set_rail_buttons(0.2, -0.5, 45)
    rk.add_motor(motor, NOZZLE_POSITION)
    rk.add_nose(length=0.610, kind="tangent", position=0.71971 + 0.610)
    rk.add_trapezoidal_fins(3, span=0.165, root_chord=0.152, tip_chord=0.0762, position=-1.04956)
    rk.add_tail(top_radius=0.203 / 2, bottom_radius=0.155 / 2, length=0.127,
                position=-1.194656)
    return rk


def fly(motor, env, drag_factor: float = 1.0, inclination: float = INCLINATION,
        heading: float = HEADING):
    """Ascent to apogee (recovery is outside the scope of this study)."""
    import warnings

    from rocketpy import Flight

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # RocketPy's ignition static-margin warning, see docs
        return Flight(rocket=rocket(motor, drag_factor), environment=env,
                      rail_length=RAIL_LENGTH, inclination=inclination, heading=heading,
                      terminate_on_apogee=True, verbose=False)


def measured_flight(data_dir: str | Path):
    """Raven altimeter record: (time [s], altitude AGL [m])."""
    import csv

    t, z = [], []
    with open(Path(data_dir) / "ndrt_2020_flight_data.csv") as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row[" Time (s)"]))
                z.append(float(row[" Altitude (Ft-AGL)"]) / 3.28084)
            except (TypeError, ValueError):
                continue
    return np.array(t), np.array(z)
