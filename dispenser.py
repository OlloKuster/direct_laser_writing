from enum import Enum

from problems.metalens._run import run as run_lens
from problems.power_splitter._run import run as run_power_splitter
from problems.frequency_filter._run import run as run_frequency_filter

class Dispenser(Enum):
    """
    Add the respective problems  here and their setup, problem and config classes as the call. Plot and data saving
    are also passed here.
    """
    LENS3D = run_lens  # Working
    POWERSPLITTER = run_power_splitter  # Working
    FFILTER = run_frequency_filter