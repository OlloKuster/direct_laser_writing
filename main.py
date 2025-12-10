import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, opt, load=None, eval=False, full_bin=False):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    run(resolution, betas, setting, opt=opt, load=load, eval=eval, full_bin=full_bin)


if __name__ == "__main__":
    init_setting = setting_loader("metalens", "normal_gauss")
    setting = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 12
    init_beta = 8
    betas = [9, 16, 32, np.inf]
    main(resolution, [init_beta], init_setting, opt="nlopt", eval=eval, full_bin=False)
    main(resolution, betas, setting, opt="optax", load=f"problems/metalens/plots/data_{init_beta}.h5", eval=eval, full_bin=True)
