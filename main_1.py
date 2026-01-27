import jax
import torch
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, load=None, eval=False, full_bin=False, run_id=0, device_id=0):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    jax_devices = jax.devices()
    jax.default_device(jax_devices[device_id])
    torch.set_default_device(f'cuda:{device_id}')

    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    init_setting = setting_loader("metalens", "normal_gauss")
    setting = setting_loader("metalens", "dlw_regular")
    eval = False
    device_id = 0
    resolution = 8
    loss_hist = []
    em_loss_hist = []
    init_beta = [16]
    betas = [1, 16, 32, np.inf]
    # loss_hist, em_loss_hist = main(resolution, init_beta, init_setting, loss_hist, em_loss_hist, opt="optax", eval=eval, full_bin=False, run_id=0, device_id=0)
    main(resolution, betas, setting, loss_hist, em_loss_hist, opt="optax", load=f"problems/metalens/plots/data_0_{init_beta[-1]}.h5", eval=eval, full_bin=False, run_id=2, device_id=1)
