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
    setting = setting_loader("metalens", "dlw_regular")
    eval = False
    device_id = 0
    resolution = 10
    loss_hist = []
    em_loss_hist = []
    betas = np.linspace(0, 2, 15)
    betas = np.append(betas, np.inf)
    run_id = 0

    target_material = [0.8, 1., 1.2, 1.4, 1.6]
    target_void = [0.8, 1., 1.2, 1.4, 1.6]
    for mat in target_material:
        for void in target_void:
            loss_hist = []
            em_loss_hist = []
            setting["target_material"] = mat
            setting["target_void"] = void
            main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=10, opt="optax", eval=eval, full_bin=False, run_id=run_id)
            run_id = run_id + 1
