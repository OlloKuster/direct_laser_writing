from enum import Enum

from problems.metalens._run import run as run_lens


class Dispenser(Enum):
    """
    Add the respective problems  here and their setup, problem and config classes as the call. Plot and data saving
    are also passed here.
    """
    LENS3D = run_lens
