# Validation against test data (planned)

Verification (`tests/`) shows the equations are solved correctly. Validation shows they are the
right equations, by comparing against measured motor firings. Each case below has published
inputs. Measured traces are digitised from the cited figures; raw data are not redistributed.

| Case | What it exercises | Inputs in hand | Status |
| --- | --- | --- | --- |
| NAWC motor 13: cylindrical port, L = 0.85 m, L/D_h = 11.2 | 0D ballistics, low erosion | burn law 0.541 cm/s at 6.9 MPa, n = 0.461 (Willcox 2007) | geometry and trace from Blomshield TP 8444 pending |
| NAWC motor 6: star-aft, L = 1.83 m, L/D_h = 69.7 | quasi-1D with erosive burning | burn law 0.678 cm/s at 6.9 MPa, n = 0.36; stations A–G (Willcox 2007, Fig. 7) | trace from Blomshield TP 8444 pending |
| Graphite throat firings, propellants S and M | throat erosion k, m | throat diameter vs time (Acharya & Kuo 2007, Figs 9–10) | digitisation pending |

Motor 6 is the headline case. Its measured pressure falls while its burning area rises, which a 0D
model cannot reproduce (Willcox et al. 2007, Figs 10 and 13).

`nawc_motors.json` records only the published burn-rate laws. Geometry is added from the primary
source, not estimated from figures.
