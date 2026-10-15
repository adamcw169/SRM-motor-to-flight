"""3D grains built from prismatic segments of a 2D cross-section.

A segment of length L with ``burning_ends`` exposed faces (0, 1 or 2) regresses as

    A_b(w) = S(w) * (L - n_e w) + n_e * (A_case - A_p(w))
    V_p(w) = (A_case - A_p(w)) * (L - n_e w)

which is exact for a prismatic section with flat end faces (a BATES segment is
the circular-port case with n_e = 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sections import CrossSection


@dataclass(frozen=True)
class GrainSegment:
    section: CrossSection
    length: float
    burning_ends: int = 0

    def __post_init__(self) -> None:
        if self.burning_ends not in (0, 1, 2):
            raise ValueError("burning_ends must be 0, 1 or 2")

    @property
    def case_area(self) -> float:
        return 0.25 * np.pi * self.section.outer_diameter**2

    @property
    def web_max(self) -> float:
        if self.burning_ends == 0:
            return self.section.web_max
        return min(self.section.web_max, self.length / self.burning_ends)

    def _len(self, w):
        return np.maximum(self.length - self.burning_ends * np.asarray(w, dtype=float), 0.0)

    def burn_area(self, w):
        w = np.asarray(w, dtype=float)
        lateral = self.section.perimeter(w) * self._len(w)
        ends = self.burning_ends * (self.case_area - self.section.port_area(w))
        return np.where(w < self.web_max, lateral + ends, 0.0)

    def propellant_volume(self, w):
        w = np.minimum(np.asarray(w, dtype=float), self.web_max)
        return (self.case_area - self.section.port_area(w)) * self._len(w)

    def envelope_volume(self) -> float:
        return self.case_area * self.length


@dataclass(frozen=True)
class Grain:
    """A stack of segments that all regress with the same web (0D assumption)."""

    segments: tuple[GrainSegment, ...]
    extra_free_volume: float = 0.0  # head-end/aft-end cavities not occupied by grain [m^3]

    @classmethod
    def single(cls, section, length, burning_ends=0, extra_free_volume=0.0) -> "Grain":
        return cls((GrainSegment(section, length, burning_ends),), extra_free_volume)

    @classmethod
    def bates(cls, section, segment_length, n_segments, extra_free_volume=0.0) -> "Grain":
        seg = GrainSegment(section, segment_length, burning_ends=2)
        return cls(tuple([seg] * n_segments), extra_free_volume)

    @property
    def web_max(self) -> float:
        return max(s.web_max for s in self.segments)

    def burn_area(self, w):
        return sum(s.burn_area(w) for s in self.segments)

    def propellant_volume(self, w):
        return sum(s.propellant_volume(w) for s in self.segments)

    def free_volume(self, w):
        envelope = sum(s.envelope_volume() for s in self.segments)
        return envelope - self.propellant_volume(w) + self.extra_free_volume


@dataclass
class AxialGrain:
    """Grain discretised into axial stations for the quasi-1D solver.

    Each station carries its own cross-section, so tapered or star-aft grains are a
    list of sections. End faces are inhibited (v0.1 limitation).
    """

    sections: list
    station_length: float
    z: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.sections)
        self.z = (np.arange(n) + 0.5) * self.station_length

    @classmethod
    def uniform(cls, section, length: float, n_stations: int = 100) -> "AxialGrain":
        return cls([section] * n_stations, length / n_stations)

    @classmethod
    def piecewise(cls, pieces: list[tuple], n_stations: int = 120) -> "AxialGrain":
        """``pieces`` = [(section, length), ...] from head end to aft end."""
        total = sum(L for _, L in pieces)
        dz = total / n_stations
        sections, edges = [], np.cumsum([0.0] + [L for _, L in pieces])
        for i in range(n_stations):
            zc = (i + 0.5) * dz
            k = int(np.searchsorted(edges, zc, side="right") - 1)
            sections.append(pieces[min(k, len(pieces) - 1)][0])
        return cls(sections, dz)

    @property
    def n(self) -> int:
        return len(self.sections)

    @property
    def length(self) -> float:
        return self.n * self.station_length

    def web_max(self) -> np.ndarray:
        return np.array([s.web_max for s in self.sections])

    def evaluate(self, webs: np.ndarray):
        """Perimeter, port area and wetted perimeter per station at per-station webs."""
        S = np.array([float(s.perimeter(w)) for s, w in zip(self.sections, webs)])
        A = np.array([float(s.port_area(w)) for s, w in zip(self.sections, webs)])
        P = np.array([float(s.wetted_perimeter(w)) for s, w in zip(self.sections, webs)])
        return S, A, P

    def propellant_volume(self, webs: np.ndarray) -> float:
        case = np.array([0.25 * np.pi * s.outer_diameter**2 for s in self.sections])
        _, A, _ = self.evaluate(webs)
        return float(np.sum((case - A) * self.station_length))

    def free_volume(self, webs: np.ndarray) -> float:
        _, A, _ = self.evaluate(webs)
        return float(np.sum(A * self.station_length))
