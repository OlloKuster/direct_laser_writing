import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from projection._projection_loader import projection_loader


def test(seed):
    np.random.seed(seed)
    rho_0 = np.random.random((50, 50, 50))
    rho_0 = gaussian_filter(rho_0, 2)
    rho_0 = np.round(rho_0)
    rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
    dose_filter = filter_loader("dose_conv", 10)

    rho_filt = dose_filter(rho_0_torch).detach().cpu().numpy()
    projection = projection_loader("tanh_jax", ConfigPrint.rho_th_GT, np.inf, 1)
    print(projection(ConfigPrint.rho_th_GT))
    rho_proj_np = projection(rho_filt)
    rho_thresh_np = rho_filt > ConfigPrint.rho_th_GT
    rho_filt_np = rho_filt

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
    ax0.imshow(rho_0[rho_0.shape[0] // 2].T, origin='lower', cmap='grey', vmin=0, vmax=1)
    ax1.imshow(rho_filt_np[rho_0.shape[0] // 2].T, origin='lower', cmap='grey')
    ax2.imshow(rho_thresh_np[rho_0.shape[0] // 2].T, origin='lower', cmap='grey', vmin=0, vmax=1)
    ax3.imshow(rho_proj_np[rho_0.shape[0] // 2].T, origin='lower', cmap='grey', vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    test(222)
