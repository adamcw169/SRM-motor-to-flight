"""Variance-based global sensitivity analysis (Sobol indices).

Sampling follows Saltelli et al. (2010): two independent scrambled-Sobol matrices
A and B (N x d) and d hybrid matrices AB_i (A with column i taken from B), for
N(d + 2) model runs. Estimators:

    first order  S_i  = mean( f_B (f_ABi - f_A) ) / Var(Y)        (Saltelli 2010)
    total order  ST_i = mean( (f_A - f_ABi)^2 ) / (2 Var(Y))      (Jansen 1999)

Confidence intervals come from bootstrap resampling of the N base rows.
``tests/test_sobol.py`` checks the estimators against the analytic indices of the
Ishigami function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.stats import qmc


@dataclass
class SobolResult:
    names: list[str]
    S1: np.ndarray
    ST: np.ndarray
    S1_ci: np.ndarray  # (d, 2) 95% bootstrap interval
    ST_ci: np.ndarray
    variance: float

    def table(self) -> str:
        rows = [f"{'input':<16}{'S1':>8}{'ST':>8}"]
        for n, s1, st in zip(self.names, self.S1, self.ST):
            rows.append(f"{n:<16}{s1:8.3f}{st:8.3f}")
        return "\n".join(rows)


def saltelli_samples(dists: Mapping, n_base: int, seed: int | None = 0):
    """Return (A, B, AB) in physical units from frozen scipy.stats distributions."""
    names = list(dists)
    d = len(names)
    u = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n_base)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    to_phys = lambda U: np.column_stack(  # noqa: E731
        [dists[n].ppf(U[:, i]) for i, n in enumerate(names)]
    )
    A, B = to_phys(u[:, :d]), to_phys(u[:, d:])
    AB = np.repeat(A[None], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    return names, A, B, AB


def sobol_indices(fA, fB, fAB, n_boot: int = 500, seed: int | None = 1):
    """First- and total-order indices from model outputs on A, B and AB_i."""
    fA, fB, fAB = np.asarray(fA), np.asarray(fB), np.asarray(fAB)

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        var = np.var(np.concatenate([a, b]), ddof=1)
        s1 = np.mean(b * (ab - a), axis=1) / var
        st = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
        return s1, st, var

    n = fA.size
    s1, st, var = est(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [est(rng.integers(0, n, n))[:2] for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boots])
    bt = np.array([b[1] for b in boots])
    ci = lambda x: np.percentile(x, [2.5, 97.5], axis=0).T  # noqa: E731
    return s1, st, ci(b1), ci(bt), var


def run_sobol(
    model: Callable[[dict], Mapping[str, float]],
    dists: Mapping,
    n_base: int = 256,
    seed: int | None = 0,
) -> dict[str, SobolResult]:
    """Sobol analysis of every scalar output returned by ``model(params) -> {name: value}``."""
    names, A, B, AB = saltelli_samples(dists, n_base, seed)
    call = lambda row: model(dict(zip(names, row)))  # noqa: E731
    outA = [call(r) for r in A]
    outB = [call(r) for r in B]
    outAB = [[call(r) for r in AB[i]] for i in range(len(names))]
    results = {}
    for key in outA[0]:
        fA = np.array([o[key] for o in outA])
        fB = np.array([o[key] for o in outB])
        fAB = np.array([[o[key] for o in row] for row in outAB])
        s1, st, c1, ct, var = sobol_indices(fA, fB, fAB)
        results[key] = SobolResult(names, s1, st, c1, ct, var)
    return results
