"""srmflight: solid rocket motor burnback, internal ballistics and UQ, coupled to 6-DOF flight.

Pipeline
--------
grain geometry  ->  internal ballistics (0D or quasi-steady 1D)  ->  motor histories
                ->  RocketPy 6-DOF flight  ->  Sobol sensitivity of motor and flight outputs
"""

from .propellant import Propellant
from .nozzle import Nozzle, ThroatErosion, thrust_coefficient
from .erosive import LenoirRobillard, MukundaPaul, NoErosion
from .results import MotorResult

__all__ = [
    "Propellant",
    "Nozzle",
    "ThroatErosion",
    "thrust_coefficient",
    "LenoirRobillard",
    "MukundaPaul",
    "NoErosion",
    "MotorResult",
]

__version__ = "0.1.0"
