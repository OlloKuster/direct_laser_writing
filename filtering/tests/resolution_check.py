import numpy as np
import scipy.ndimage
import jax
import matplotlib.pyplot as plt
import scipy

from filtering._filter_loader import filter_loader
from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from filtering.dose_model.utils_dose_sim import calc_psf_GT
from projection.SSP.subpixel_smoothed_projection import ssp_proj_jax_f


def test(seed):
    jax.config.update("jax_enable_x64", True)
    np.random.seed(seed)
    resolution = 15
    psf = calc_psf_GT(resolution)
    for factor in np.linspace(0, 1, 20):
        rho_0 = np.ones((5*resolution, 5*resolution, 5*resolution))
        rho_0[:, :rho_0.shape[1]//2] = 0
        dose_filter = filter_loader("dose_conv", resolution, 0.05*factor, psf)

        rho_filt = dose_filter(rho_0)
        result = np.array(rho_filt).squeeze()
        result = np.where(result > ConfigPrint.rho_th_GT, 1, 0)
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(rho_0[rho_0.shape[0]//2])
        ax[1].imshow(result[result.shape[0]//2])
        plt.savefig(f"plots/rho_{factor}.png")
        plt.close()
    return

if __name__ == "__main__":
    test(42342)