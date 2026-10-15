"""2D grain cross-sections: burning perimeter and port area as functions of burnt web.

A cross-section is the propellant annulus between a port (the gas passage) and the
case. Every section exposes

    perimeter(w)       burning perimeter S(w) [m]
    port_area(w)       port (gas) area A_p(w) [m^2]
    wetted_perimeter(w) gas-wetted perimeter, burning surface + exposed case [m]
    web_max            web at which the last propellant reaches the case [m]

Uniform regression of a section is a pure offset of the port boundary, so the
burning surface at web w is the w-level set of the signed distance to the initial
port (Willcox et al. 2007; Ren et al. 2021). ``DistanceFieldSection`` computes that
distance on a Cartesian grid and extracts the level sets with marching squares.
In a quasi-1D motor the burn rate varies only along the axis, so each axial
station regresses uniformly and needs only its own web - no level-set PDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree


class CrossSection(Protocol):
    outer_diameter: float

    @property
    def web_max(self) -> float: ...

    def perimeter(self, w): ...

    def port_area(self, w): ...

    def wetted_perimeter(self, w): ...


def _case_area(outer_diameter: float) -> float:
    return 0.25 * np.pi * outer_diameter**2


@dataclass(frozen=True)
class CircularPort:
    """Centre-perforated (tubular) section; exact closed form."""

    outer_diameter: float
    port_diameter: float

    def __post_init__(self) -> None:
        if not 0.0 < self.port_diameter < self.outer_diameter:
            raise ValueError("need 0 < port_diameter < outer_diameter")

    @property
    def web_max(self) -> float:
        return 0.5 * (self.outer_diameter - self.port_diameter)

    @property
    def case_area(self) -> float:
        return _case_area(self.outer_diameter)

    def _r(self, w):
        return np.minimum(0.5 * self.port_diameter + np.asarray(w, dtype=float),
                          0.5 * self.outer_diameter)

    def perimeter(self, w):
        w = np.asarray(w, dtype=float)
        return np.where(w < self.web_max, 2.0 * np.pi * self._r(w), 0.0)

    def port_area(self, w):
        return np.pi * self._r(w) ** 2

    def wetted_perimeter(self, w):
        return 2.0 * np.pi * self._r(w)


@dataclass
class DistanceFieldSection:
    """Arbitrary port shape regressed through a tabulated signed-distance field.

    Parameters
    ----------
    outer_diameter : case (propellant outer) diameter [m]
    port_polygon : (N, 2) closed or open polygon of the initial port boundary [m]
    resolution : grid points across the case diameter
    n_levels : number of web levels tabulated between 0 and web_max
    """

    outer_diameter: float
    port_polygon: np.ndarray
    resolution: int = 401
    n_levels: int = 241
    _web: np.ndarray = field(init=False, repr=False)
    _perim: np.ndarray = field(init=False, repr=False)
    _area: np.ndarray = field(init=False, repr=False)
    _wet: np.ndarray = field(init=False, repr=False)
    _web_max: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        poly = np.asarray(self.port_polygon, dtype=float)
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        self.port_polygon = poly
        R = 0.5 * self.outer_diameter
        if np.max(np.hypot(poly[:, 0], poly[:, 1])) >= R:
            raise ValueError("port polygon must lie inside the case")
        self._build(poly, R)

    # ------------------------------------------------------------------ build
    def _build(self, poly: np.ndarray, R: float) -> None:
        n = self.resolution
        x = np.linspace(-R, R, n)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, x)
        pts = np.column_stack([X.ravel(), Y.ravel()])

        # Densely resampled boundary -> nearest-point distance; error ~ h^2 / (8 d).
        boundary = _resample_closed(poly, spacing=dx / 8.0)
        tree = cKDTree(boundary)
        dist, _ = tree.query(pts)
        inside = Path(np.vstack([poly, poly[:1]])).contains_points(pts)
        phi = np.where(inside, -dist, dist).reshape(X.shape)

        # Last propellant to burn sits on the case wall.
        theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        circle = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
        d_case, _ = tree.query(circle)
        self._web_max = float(np.max(d_case))

        webs = np.linspace(0.0, self._web_max, self.n_levels)
        gen = contour_generator(x=x, y=x, z=phi)
        perim = np.array([_clipped_length(gen.lines(w), R) for w in webs])
        perim[-1] = 0.0
        # Exposed case arc: fraction of the wall the flame has reached.
        exposed = np.array([np.mean(d_case <= w) for w in webs]) * 2.0 * np.pi * R

        a0 = abs(_shoelace(poly))
        area = a0 + cumulative_trapezoid(perim, webs, initial=0.0)
        area = np.minimum(area, np.pi * R**2)

        self._web, self._perim, self._area = webs, perim, area
        self._wet = perim + exposed
        self._phi, self._grid = phi, x

    # ---------------------------------------------------------------- queries
    @property
    def web_max(self) -> float:
        return self._web_max

    @property
    def case_area(self) -> float:
        return _case_area(self.outer_diameter)

    def perimeter(self, w):
        w = np.asarray(w, dtype=float)
        s = np.interp(w, self._web, self._perim)
        return np.where(w < self._web_max, s, 0.0)

    def port_area(self, w):
        return np.interp(w, self._web, self._area, right=self.case_area)

    def wetted_perimeter(self, w):
        return np.interp(w, self._web, self._wet, right=np.pi * self.outer_diameter)

    def contours(self, w: float):
        """Burning-surface polylines at web ``w`` (for plotting)."""
        gen = contour_generator(x=self._grid, y=self._grid, z=self._phi)
        return gen.lines(w)


# ---------------------------------------------------------------- shape helpers
def star_polygon(n_points: int, r_tip: float, r_valley: float,
                 samples_per_edge: int = 40) -> np.ndarray:
    """Straight-sided star port with ``n_points`` tips at r_tip and valleys at r_valley."""
    if not 0.0 < r_valley < r_tip:
        raise ValueError("need 0 < r_valley < r_tip")
    angles = np.arange(2 * n_points) * np.pi / n_points + 0.5 * np.pi
    radii = np.where(np.arange(2 * n_points) % 2 == 0, r_tip, r_valley)
    verts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _resample_closed(verts, n_per_edge=samples_per_edge)


def circle_polygon(radius: float, n: int = 720) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def square_polygon(side: float) -> np.ndarray:
    h = 0.5 * side
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])


def _resample_closed(poly: np.ndarray, spacing: float | None = None,
                     n_per_edge: int | None = None) -> np.ndarray:
    nxt = np.roll(poly, -1, axis=0)
    out = []
    for p, q in zip(poly, nxt):
        L = np.hypot(*(q - p))
        k = n_per_edge if n_per_edge is not None else max(int(np.ceil(L / spacing)), 1)
        s = np.arange(k)[:, None] / k
        out.append(p + s * (q - p))
    return np.vstack(out)


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clipped_length(lines, R: float) -> float:
    """Total length of polylines, counting only segments whose midpoint lies in the case."""
    total = 0.0
    for line in lines:
        if len(line) < 2:
            continue
        seg = np.diff(line, axis=0)
        mid = 0.5 * (line[1:] + line[:-1])
        keep = np.hypot(mid[:, 0], mid[:, 1]) < R
        total += float(np.sum(np.hypot(seg[keep, 0], seg[keep, 1])))
    return total
