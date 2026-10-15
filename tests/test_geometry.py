"""Burnback verification against closed-form offsets."""

import numpy as np
import pytest

from srmflight.geometry import (
    CircularPort,
    DistanceFieldSection,
    GrainSegment,
    circle_polygon,
    square_polygon,
    star_polygon,
)

OD = 0.1


def test_circular_port_closed_form():
    s = CircularPort(OD, 0.04)
    assert s.web_max == pytest.approx(0.03)
    assert s.perimeter(0.01) == pytest.approx(2 * np.pi * 0.03)
    assert s.port_area(0.01) == pytest.approx(np.pi * 0.03**2)
    assert s.perimeter(0.031) == 0.0


@pytest.mark.parametrize("w", [0.0, 0.005, 0.015, 0.028])
def test_distance_field_reproduces_tube(w):
    num, ref = DistanceFieldSection(OD, circle_polygon(0.02)), CircularPort(OD, 0.04)
    assert num.perimeter(w) == pytest.approx(ref.perimeter(w), rel=1e-3)
    assert num.port_area(w) == pytest.approx(ref.port_area(w), rel=1e-3)
    assert num.web_max == pytest.approx(ref.web_max, rel=1e-3)


@pytest.mark.parametrize("w", [0.002, 0.008, 0.015])
def test_convex_offset_perimeter(w):
    """Offset of a convex polygon: S(w) = S0 + 2 pi w (Steiner formula)."""
    side = 0.03
    s = DistanceFieldSection(OD, square_polygon(side))
    assert s.perimeter(w) == pytest.approx(4 * side + 2 * np.pi * w, rel=2e-3)


@pytest.mark.parametrize("w", [0.004, 0.012, 0.024])
def test_port_area_independent_of_perimeter(w):
    """Port area from integrating S(w) agrees with direct cell counting of phi <= w."""
    s = DistanceFieldSection(OD, star_polygon(6, 0.03, 0.015), resolution=801)
    x = s._grid
    X, Y = np.meshgrid(x, x)
    in_case = np.hypot(X, Y) <= 0.5 * OD
    counted = np.sum((s._phi <= w) & in_case) * (x[1] - x[0]) ** 2
    assert s.port_area(w) == pytest.approx(counted, rel=5e-3)


def test_star_grid_convergence():
    """Perimeter error against a 1001-point reference falls as the grid is refined."""
    star = star_polygon(6, 0.03, 0.015)
    w = 0.01
    ref = DistanceFieldSection(OD, star, resolution=1001).perimeter(w)
    err = [abs(DistanceFieldSection(OD, star, resolution=n).perimeter(w) - ref) / ref
           for n in (201, 401)]
    assert err[1] < 0.5 * err[0]
    assert err[1] < 1e-3


def test_bates_segment_end_burning():
    sec, L = CircularPort(OD, 0.04), 0.12
    seg = GrainSegment(sec, L, burning_ends=2)
    w = 0.01
    lateral = 2 * np.pi * (0.02 + w) * (L - 2 * w)
    ends = 2 * np.pi * (0.05**2 - (0.02 + w) ** 2)
    assert seg.burn_area(w) == pytest.approx(lateral + ends)


def test_volume_balance():
    """V_p(0) - V_p(w) equals the integral of A_b over web."""
    seg = GrainSegment(DistanceFieldSection(OD, star_polygon(6, 0.03, 0.015)), 0.3, 1)
    w = np.linspace(0.0, 0.02, 400)
    burnt = np.trapezoid(seg.burn_area(w), w)
    assert seg.propellant_volume(0.0) - seg.propellant_volume(0.02) == pytest.approx(burnt,
                                                                                    rel=5e-3)
