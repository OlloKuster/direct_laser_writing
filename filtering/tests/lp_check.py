import numpy as np
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
import scipy

from filtering._filter_loader import filter_loader
from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from projection.SSP.subpixel_smoothed_projection import ssp_proj_jax_f


def test(seed):
    np.random.seed(seed)
    resolution = 10
    for factor in np.linspace(0., 0.01, 101):
        ConfigPrint.lp = factor
        rho_0 = np.zeros((5*resolution, 5*resolution, 5*resolution))
        rho_0[rho_0.shape[0]//4:-rho_0.shape[0]//4, rho_0.shape[1]//4:-rho_0.shape[1]//4,  rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1

        rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
        dose_filter = filter_loader("dose_conv", resolution)

        rho_filt = dose_filter(rho_0_torch)
        result = rho_filt.detach().cpu().numpy().squeeze()
        result = np.where(result > ConfigPrint.rho_th_GT, 1, 0)
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(rho_0[rho_0.shape[0]//2])
        ax[1].imshow(result[result.shape[0]//2])
        plt.savefig(f"plots_1/lp_{factor:3f}.png")
        plt.close()
        if np.sum(result) > 0:
            print(factor)
            break
    return

if __name__ == "__main__":
    test(42342)