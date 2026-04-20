import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False,
         run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval,
               full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_regular")
    eval = True
    device_id = 0
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas = [np.inf]
    run_id = 0

    target_material = [-0.99, -0.8, -0.6, -0.4, -0.2]
    target_void = [-0.99, -0.8, -0.6, -0.4, -0.2]
    for mat in target_material:
        for void in target_void:
            loss_hist = []
            em_loss_hist = []
            setting["target_material"] = mat
            setting["target_void"] = void
            loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=15,
                                           opt="nlopt", eval=eval, full_bin=False, run_id=run_id,
                                           load=f'/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/base/data_base_no_filter.h5')
            run_id = run_id + 1
