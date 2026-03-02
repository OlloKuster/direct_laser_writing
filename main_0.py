import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = setting["run"]

    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval, full_bin=full_bin, run_id=run_id)



if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas = [16, 32, np.inf]
    loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=15, opt="nlopt", eval=eval, full_bin=False, run_id=0)
