# Validation: NDRT 2020 flight

Verification (`tests/`, [`docs/VERIFICATION.md`](../docs/VERIFICATION.md)) shows the equations are
solved correctly. Validation compares against measured data from a real flight: the Notre Dame
Rocket Team launch of 23 Feb 2020, RocketPy's published validation case (Ceotto et al. 2021).

```bash
python validation/ndrt/fetch_data.py         # certified thrust curve, Raven record, ERA5 file
python examples/03_ndrt_calibration.py       # -> ndrt/calibration.json
python examples/04_ndrt_flight_validation.py # -> ndrt/flight_validation.json
python examples/05_ndrt_flight_sobol.py      # -> ndrt/sobol_results.json (about 20 min)
```

The data are downloaded from the RocketPy repository and not redistributed here.

## 1. Motor: Cesaroni L1395 against its certified thrust curve

The model uses five BATES grains with the published geometry and loaded mass. The burn-rate
coefficient, c* and throat-erosion rate are fitted at each pressure exponent across Davenas'
range for non-aluminised composites.

| n | a at 6.9 MPa (mm/s) | c* (m/s) | k at 6.9 MPa (mm/s) | RMS thrust error | impulse error |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 7.49 | 1436 | 1.26 | 1.8 % | +0.3 % |
| 0.40 | 8.10 | 1434 | 0.89 | 2.0 % | +0.5 % |
| 0.50 | 8.74 | 1436 | 0.67 | 2.5 % | +0.9 % |

- c* is identified to ±0.1 %. n and k trade off, since a regressive trace can come from a low
  exponent or from throat growth.
- Every fitted k sits inside the 0.25–1.3 mm/s band that Acharya & Kuo (2007, Fig. 4a) predict for
  non-aluminised propellant on graphite at 7 MPa.
- The ignition spike (first 0.25 s) and the tail-off are not modelled (0D, simultaneous burnout).

## 2. Flight: apogee against the Raven altimeter

| Motor input to RocketPy | Apogee AGL (m) | Error |
| --- | --- | --- |
| Measured (Raven) | 1320.4 | — |
| Certified thrust curve | 1309.7 | −0.81 % |
| srmflight motor, n = 0.30 | 1314.0 | −0.48 % |
| srmflight motor, n = 0.40 | 1317.3 | −0.23 % |
| srmflight motor, n = 0.50 | 1323.3 | +0.22 % |

- Airframe and launch parameters are RocketPy's acceptance-test values; the atmosphere is ERA5.
- The certified-curve run reproduces RocketPy's published −0.75 % to within 0.06 points.
- The model's slightly closer match is within its own calibration uncertainty (its impulse is
  0.3–0.9 % above the certificate), so treat the two as equivalent rather than one as better.
- RocketPy's constant-exhaust-velocity mass flow differs from the ballistics mass flow by at most
  0.5 % of propellant mass.
- RocketPy reports a −0.18 cal static margin at ignition for this configuration from its own
  aerodynamic estimate (the team measured +2.9 cal). The ascent is near-vertical and apogee still
  matches, so the configuration is kept identical to the published case.

## 3. Sensitivity: which uncertainties move the flight?

There are 5,120 motor + 6-DOF flights (Saltelli design, N = 512). Total-order indices for apogee
(σ = 37.5 m):

| Input | Distribution | Source | S_T |
| --- | --- | --- | --- |
| Grain temperature | U(−45, +60) °C | Davenas ch. 1 §2.6 | 0.53 |
| Burn-rate deviation r/r₀ | U(0.95, 1.05) | Davenas ch. 4 §3.1.2 | 0.21 |
| Drag coefficient | N(1, 0.033) × C_D | RocketPy NDRT case | 0.15 |
| Wind speed | N(1, 0.30) × ERA5 | RocketPy NDRT case | 0.06 |
| c* scatter | N(1, 0.0037) | Davenas ch. 3 (σ_Isp 0.5–1 s) | 0.05 |
| Temperature sensitivity π_K | U(0.15, 0.35) %/°C | around Davenas' typical 0.25 | 0.04 |
| Pressure exponent n | U(0.30, 0.50), calibrated family | Davenas ch. 2 Table 3 | 0.006 |
| Rail tilt | half-normal, σ = 1° | RocketPy NDRT case | 0.006 |

- **Motor inputs carry about 80 % of the apogee variance.** For max speed and rail-exit speed it is
  over 99 %.
- **The calibration ambiguity is harmless.** The pressure exponent, the one parameter the certified
  curve cannot pin down, contributes under 1 %.
- S₁ ≈ S_T for every input, so the inputs act almost additively.
- The temperature envelope is a qualification-style stress range, not a launch-day spread. The
  deterministic sweep in `examples/06_ndrt_temperature.py` shows apogee rising by 89 m from −45 to
  +60 °C.
