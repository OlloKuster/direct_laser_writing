import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, eval):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    run(resolution, betas, setting, eval=eval)


if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_regular")
    main(12, [8, 16, 32, np.inf], setting, eval=False)
