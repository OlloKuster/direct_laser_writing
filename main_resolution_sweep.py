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
    setting = setting_loader("metalens", "gauss_em_only")
    eval = False
    resolution = 10
    betas = [32]
    run_id = 0

    resolutions = [16]
    for resolution in resolutions:
        loss_hist = []
        em_loss_hist = []
        loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=20, opt="nlopt", eval=eval, full_bin=False, run_id=7)

        run_id = run_id + 1
