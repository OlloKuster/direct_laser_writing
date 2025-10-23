from enum import Enum

from problems.metalens._run import run as run_lens
from problems.mode_converter_jax._run import run as run_mode_converter

class Dispenser(Enum):
    """
    Add the respective problems  here and their setup, problem and config classes as the call. Plot and data saving
    are also passed here.
    """
    LENS3D = run_lens
    MODECONVERTER = run_mode_converter
