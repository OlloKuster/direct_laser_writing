import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint


def test(seed):
    np.random.seed(seed)
    rho_0 = np.random.random((50, 50, 50))
    rho_0 = gaussian_filter(rho_0, 5)
    rho_0 = np.round(rho_0)
    dose_filter = filter_loader("dose_conv", 10)

    rho_filt = dose_filter(torch.tensor(rho_0, device='cuda'))
    rho_filt = rho_filt > ConfigPrint.rho_th_GT
    rho_filt = rho_filt.detach().cpu().numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(rho_0[rho_0.shape[0]//2].T, origin='lower', cmap='binary')
    ax1.imshow(rho_filt[rho_0.shape[0]//2].T, origin='lower', cmap='binary')
    plt.show()


if __name__ == "__main__":
    test(123)