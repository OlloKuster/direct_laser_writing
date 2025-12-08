import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, load=None, eval=False):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    run(resolution, betas, setting, load=load, eval=eval)


if __name__ == "__main__":
    init_setting = setting_loader("metalens", "normal_gauss")
    setting = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 8
    init_beta = 8
    betas = [8, 16, 32, np.inf]
    main(resolution, betas, setting, eval=eval)
