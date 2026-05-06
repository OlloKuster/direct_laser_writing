from enum import Enum

from problems.metalens._run import run as run_lens
from problems.mode_converter._run import run as run_mode_converter
from problems.multiplexer._run import run as run_multiplexer
from problems.polarization_splitter._run import run as run_pol_splitter
from problems.power_splitter._run import run as run_power_splitter

class Dispenser(Enum):
    """
    Add the respective problems  here and their setup, problem and config classes as the call. Plot and data saving
    are also passed here.
    """
    LENS3D = run_lens  # Working
    MODECONVERTER = run_mode_converter  # Experimental
    MULTIPLEXER = run_multiplexer  # Experimental
    POLSPLITTER = run_pol_splitter  # Experimental
    POWERSPLITTER = run_power_splitter  # Working