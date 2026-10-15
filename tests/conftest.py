import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from srmflight import Nozzle, Propellant  # noqa: E402


@pytest.fixture
def propellant():
    """Generic non-aluminised AP/HTPB-class propellant (illustrative values)."""
    return Propellant(density=1750.0, a_ref=0.007, n=0.35, c_star=1500.0, gamma=1.2,
                      sigma_p=0.002)


@pytest.fixture
def nozzle():
    return Nozzle(throat_diameter=0.02, exit_diameter=0.05)
