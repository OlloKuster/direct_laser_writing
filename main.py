import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, load=None, eval=False, full_bin=False, run_id=0):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    init_setting = setting_loader("metalens", "normal_gauss")
    setting = setting_loader("metalens", "dlw_regular")
    eval = False
    resolution = 16
    loss_hist = []
    em_loss_hist = []
    init_beta = [16, 32, 64]
    betas = [np.inf]
    loss_hist, em_loss_hist = main(resolution, init_beta, init_setting, loss_hist, em_loss_hist, opt="optax", eval=eval, full_bin=False, run_id=0)
    main(resolution, betas, setting, loss_hist, em_loss_hist, opt="optax", load=f"problems/metalens/plots/data_0_{init_beta[-1]}.h5", eval=eval, full_bin=False, run_id=1)
