import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = Dispenser.MODECONVERTER
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting = setting_loader("mode_converter", "dlw_regular")
    eval = True
    resolution = 10
    loss_hist = []
    em_loss_hist = []
    betas = np.logspace(0, 2, 15)
    betas = np.append(betas, np.inf)
    main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=10, opt="optax", eval=eval, full_bin=False, run_id=0)
