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
    eval = True
    resolution = 14
    betas = [16, 32, np.inf]
    run_id = 0

    feature_size_factor = [1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1]
    for fact in feature_size_factor:
        loss_hist = []
        em_loss_hist = []
        setting["filter_factor"] = fact / np.sqrt(3)
        loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=15, opt="nlopt", eval=eval, full_bin=False, run_id=run_id)

        run_id = run_id + 1
    # main(resolution, betas, setting, loss_hist, em_loss_hist, opt="optax", load=f"problems/metalens/plots/data_0_{init_beta[-1]}.h5", eval=eval, full_bin=False, run_id=2, device_id=1)
