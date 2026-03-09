import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_robust")
    eval = False
    device_id = 0
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas = [8, 16, np.inf]
    run_id = 0

    lps = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    for lp in lps:
        loss_hist = []
        em_loss_hist = []
        setting["lp_deviation"] = lp
        loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=15,
                                       opt="nlopt", eval=eval, full_bin=False, run_id=run_id)
        run_id = run_id + 1
