# Verification

Verification asks whether the code solves its equations correctly. Every check below has a closed
form, a conserved quantity or an analytic benchmark. Each is also a unit test in `tests/`, so a
regression fails CI.

Reproduce with `python examples/00_verification_report.py`.

| Check | Metric | Result | Tolerance |
| --- | --- | --- | --- |
| Tube burnback, distance field vs exact perimeter | max rel. error | 7.2e-05 | 1e-03 |
| Square port, offset perimeter S₀ + 2πw (Steiner) | max rel. error | 1.9e-04 | 2e-03 |
| Star perimeter, 401 vs 1001 grid | rel. difference | 1.9e-04 | 1e-03 |
| 0D burn time vs closed-form progressive tube | rel. error | 3.4e-03 | 1e-02 |
| 0D mass closure (expelled / loaded) | deviation from 1 | 1.9e-04 | 2e-03 |
| Quasi-1D mass closure, Δw = web/400 | deviation from 1 | 3.5e-04 | 1e-03 |
| Quasi-1D → 0D limit (L/D = 2.5), total impulse | rel. difference | 1.1e-03 | 1e-02 |
| Sobol indices, Ishigami function (N = 4096) | max abs. error | 9.2e-04 | 3e-02 |

## Notes

- **Closed-form tube.** With inhibited ends, A_b ∝ (r_p + w) and quasi-steady pressure give
  dw/dt = c (r_p + w)^(n/(1−n)), which integrates exactly. The 0.34% residual is the gas-storage
  term the closed form neglects (ρ_g/ρ_p ≈ 0.5%).
- **Quasi-1D mass closure** converges at first order in the web step (1.7e-3 at web/100, 3.5e-4 at
  web/400), as expected for explicit web advance.
- **Ishigami** (a = 7, b = 0.1) is the standard benchmark for Sobol estimators. Its analytic indices
  are S₁ = (0.314, 0.442, 0) and S_T = (0.558, 0.442, 0.244).

Validation against measured firings is tracked separately in
[`validation/README.md`](../validation/README.md).
