import numpy as np
import scipy.ndimage
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint


def test(seed):
    np.random.seed(seed)
    factors = [1, 2, 3, 4, 5]
    result = []
    resolution = 5
    for factor in factors:
        rho_0_init = np.random.random((5*resolution, 5*resolution, 5*resolution))
        rho_0_g = gaussian_filter(rho_0_init, 2)
        rho_0_scaled = scipy.ndimage.zoom(rho_0_g, factor)
        rho_0 = np.round(rho_0_scaled)
    # rho_0 = np.ones((5*resolution, 5*resolution, 5*resolution))
        rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
        dose_filter = filter_loader("dose_conv", resolution * factor)

        rho_filt = dose_filter(rho_0_torch)
        result.append(rho_filt.detach().cpu().numpy())
    plt.plot(factors, result)
    plt.show()

if __name__ == "__main__":
    test(42342)