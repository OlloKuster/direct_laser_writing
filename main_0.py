import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 20
    loss_hist = []
    em_loss_hist = []
    betas = [1, 16, 32, np.inf]
    main(resolution, betas, setting, loss_hist, em_loss_hist, opt="nlopt", eval=eval, full_bin=True, run_id=1)
