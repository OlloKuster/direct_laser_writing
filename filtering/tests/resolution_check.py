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
    factors = [1, 2, 3, 4, 5]
    result = []
    resolution = 5
    rho_0_init = np.random.random((5*resolution, 5*resolution, 5*resolution))
    for factor in factors:
        rho_0_g = scipy.ndimage.gaussian_filter(rho_0_init, 3)
        rho_0_scaled = scipy.ndimage.zoom(rho_0_g, factor)
        rho_0 = np.round(rho_0_scaled)
        rho_0 = np.ones((5*resolution*factor, 5*resolution*factor, 5*resolution*factor))
        rho_0[:, :rho_0.shape[1]//2] = 0
        rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
        dose_filter = filter_loader("dose_conv", resolution * factor)

        rho_filt = dose_filter(rho_0_torch)
        result = rho_filt.detach().cpu().numpy().squeeze()
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(rho_0[rho_0.shape[0]//2])
        ax[1].imshow(result[result.shape[0]//2])
        plt.savefig(f"plots/rho_{factor}.png")
        plt.close()
    return

    resolutions = np.arange(10, 20, 1)
    print(resolutions)
    max = []
    for resolution in resolutions:
        resolution = int(resolution)
        print(f"Resolution {resolution}")
        dose_filter = dose_filter_f(resolution)

        projection = ssp_proj_jax_f(ConfigPrint.rho_th_GT, 8, resolution)

        rho_0_0 = np.random.rand(5*resolution, 5*resolution, 5*resolution)
        rho_0 = np.round(scipy.ndimage.gaussian_filter(rho_0_0, sigma=1*resolution))
        rho_0 = np.ones_like(rho_0) * ConfigPrint.rho_th_GT
        rho = dose_filter(torch.tensor(rho_0, device='cuda', requires_grad=True))
        #
        rho = rho.detach().cpu().numpy().squeeze()
        rho_proj = projection(rho)
        # f, ax = plt.subplots(1, 3)
        # ax[0].imshow(rho_0[rho_proj.shape[0]//2])
        # ax[1].imshow(rho[rho_proj.shape[0]//2])
        # ax[2].imshow(rho_proj[rho_proj.shape[0]//2], vmin=0, vmax=1)
        # plt.show()
        max.append(np.max(rho))
        print(f"Resolution {resolution}; Max {np.max(rho)}")

    plt.plot(resolutions, max)
    plt.ylim(0, 1)
    plt.hlines(y=0.5, xmin=resolutions[0], xmax=resolutions[-1])
    plt.show()
    # psf = psf.detach().cpu().numpy().squeeze()

if __name__ == "__main__":
    test(42342)