# Theory

Every model in `srmflight` is a published, reduced-order method. This page states each one, the
assumptions it carries and where it lives in the code.

## 1. Grain burnback — `geometry/`

Propellant regresses normal to its surface at the local burn rate. For uniform regression of a
prismatic section, the burning surface at web *w* is the *w*-level set of the signed distance φ
to the initial port boundary (minimum distance function; Willcox et al. 2007, Ren et al. 2021):

```
Γ(w) = { x : φ(x) = w },   S(w) = |Γ(w) ∩ case|,   A_p(w) = A_p(0) + ∫₀ʷ S dw'
```

`DistanceFieldSection` evaluates φ on a Cartesian grid from a densely resampled boundary
(k-d tree nearest point, error ≈ h²/8d), extracts level sets with marching squares and clips
them at the case wall. Tabulating S(w) and A_p(w) once makes every later evaluation an
interpolation.

**Why no level-set PDE.** Ren et al. show the MDF offset is exact only for spatially uniform burn
rate; erosive burning makes the rate non-uniform, which normally forces a level-set PDE. In a
quasi-1D motor, however, the rate varies only along the axis. Each axial station's cross-section
therefore burns uniformly and needs only its own web wⱼ. The 3D problem becomes one table per
station plus a vector of webs.

A segment of length L with nₑ burning end faces:

```
A_b(w) = S(w)(L − nₑw) + nₑ(A_case − A_p(w))
```

## 2. Burn rate — `propellant.py`, `erosive.py`

Saint Robert's law with exponential temperature sensitivity:

```
r₀ = a_ref · exp(σ_p (Tᵢ − T_ref)) · (p / p_ref)ⁿ
```

**Lenoir–Robillard (1957)** adds a convective term from a transpiration-corrected Chilton–Colburn
heat balance. It is implicit in r and solved by bracketed root finding:

```
r = r₀ + α G^0.8 / L^0.2 · exp(−β ρ_p r / G),   β ≈ 53
α = 0.0288 c_p μ^0.2 Pr^(−2/3) / (ρ_p c_s) · (T_f − T_s) / (T_s − Tᵢ)
```

L is the hydraulic diameter by default (Lawrence et al. 1968), or the head-end distance as in the
original paper. Because α ∝ 1/(T_s − Tᵢ), one grain-temperature sample shifts the base and
erosive rates together. The UQ study keeps that coupling.

**Mukunda–Paul (1997)**, the universal correlation:

```
η = r/r₀ = 1 + 0.023 (g^0.8 − g_th^0.8) H(g − g_th)
g = g₀ (Re₀/1000)^−0.125,   g₀ = G/(ρ_p r₀),   Re₀ = ρ_p r₀ d / μ,   g_th = 35
```

## 3. Internal ballistics — `ballistics/`

**0D** (`zerod.py`): an isothermal chamber at the delivered c*, with R T₀ = (c* Γ)², where
Γ is the Vandenkerckhove function:

```
V/(R T₀) dp/dt = A_b r (ρ_p − p/(R T₀)) − p A_t / c*
```

Integrated with LSODA from the equilibrium start to burnout. Blowdown follows from the same
equation with A_b = 0.

**Quasi-steady quasi-1D** (`quasi1d.py`): steady inviscid port flow with normal mass injection,
marched from the head end (M = 0):

```
ṁⱼ₊₁ = ṁⱼ + ρ_p rⱼ Sⱼ Δz
(pA + ṁu)ⱼ₊₁ = (pA + ṁu)ⱼ + pⱼ (Aⱼ₊₁ − Aⱼ)
T = T₀ / (1 + (γ−1)/2 M²)
```

The head-end pressure is the root of ṁ_N = p₀,N A_t / c*. Webs advance with per-station rates,
and the step is limited so that no station overshoots burnout.

Neglected: gas storage in the 1D flow, ignition and flame spreading, Zel'dovich–Novozhilov
dynamic burning, two-phase flow, and grain deformation. Willcox et al. (2007) find dynamic burning
decays within about 0.15 s in the motors they studied, so it mainly affects the ignition spike.

## 4. Nozzle and throat erosion — `nozzle.py`

Ideal thrust coefficient times an efficiency factor. Throat recession follows
`dr_t/dt = k (p/p_ref)^m`, with m = 1 following the near-linear pressure dependence reported by
Acharya & Kuo (2007). k is a calibration and uncertainty parameter, not a physical constant.
For non-aluminised AP/HTPB on graphite, their kinetic schemes predict roughly 0.25–1.3 mm/s at
7 MPa (Acharya & Kuo 2007, Fig. 4a), which bounds what a calibration may return.

Temperature sensitivity uses Davenas' constant-K_n coefficient π_K (about 0.25 %/°C for
composites) converted to the constant-pressure form σ_p = (1 − n) π_K (Davenas 1993, ch. 1 §2.6).

## 5. Calibration to a certified thrust curve — `calibration.py`

Commercial motors publish grain geometry and a static-test thrust curve, not their propellant
data. With η_F fixed, the unknowns are identifiable from the trace:

```
I ≈ η_F C_F c* m_p   →  c*          F ≈ η_F C_F p_c A_t   →  chamber pressure
t_b ≈ w / (a p_c^n)  →  a           curve shape            →  n and throat erosion k
```

`fit_thrust_curve` solves for the parameters by bounded least squares on the quasi-steady part
of the trace plus a total-impulse constraint. For the Cesaroni L1395, c* is identified to
±0.1 % but n and k trade off: a regressive trace can come from a low exponent or from throat
growth. `cases/ndrt2020.calibrate` therefore fits (a, c*, k) at each n across Davenas' range
for non-aluminised composites (0.30–0.50) and the uncertainty study samples along that family.

## 6. Flight — `flight.py`, `cases/ndrt2020.py`

Thrust histories pass to a RocketPy `SolidMotor` with the real grain geometry (Ceotto et al.
2021). RocketPy derives mass flow from a constant exhaust velocity; `exhaust_velocity_error`
quantifies how far that sits from the ballistics mass flow. The atmosphere is the ERA5
reanalysis for the launch, read the same way as RocketPy's reanalysis loader (nearest time,
bilinear in latitude and longitude, geopotential to geometric height).

## 7. Global sensitivity — `uq/sobol.py`

Saltelli (2010) sampling on scrambled Sobol sequences, giving N(d + 2) runs. First-order indices
use the Saltelli 2010 estimator, total-order indices the Jansen 1999 estimator, and confidence
intervals come from bootstrapping. S₁ is the variance explained by an input alone; S_T includes
all its interactions.
