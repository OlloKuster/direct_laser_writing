import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy
import jax

from filtering._filter_loader import filter_loader
from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from filtering.dose_model.utils_dose_sim import calc_psf_GT
from projection.SSP.subpixel_smoothed_projection import ssp_proj_jax_f
from projection._projection_loader import projection_loader


def test(seed):
    jax.config.update("jax_enable_x64", True)

    np.random.seed(seed)
    resolution = 6
    ConfigPrint.lp = ConfigPrint.lp
    rho_0 = np.ones((5*resolution, 5*resolution, 5*resolution))
    rho_0 = np.round(scipy.ndimage.gaussian_filter(np.random.rand(rho_0.shape[0],
                                                                  rho_0.shape[1],
                                                                  1), sigma=0.25 * resolution))

    rho_0 = np.repeat(rho_0, rho_0.shape[2] * resolution, axis=2)
    psf = calc_psf_GT(resolution)
    i = 0
    f, ax = plt.subplots(1, 3)
    for threshold_value in [0.4, 0.5, 0.6]:
        mask = np.ones_like(rho_0)
        mask[:int(1 * resolution)] = 0
        mask[:, :int(1 * resolution)] = 0
        mask[:, :, -int(1 * resolution):] = 0
        # rho_0 = rho_0 * mask
        # rho_0[rho_0.shape[0]//4:-rho_0.shape[0]//4, rho_0.shape[1]//4:-rho_0.shape[1]//4,  rho_0.shape[2]//4:-rho_0.shape[2]//4] = 0.4

        # rho_0 = np.round(scipy.ndimage.gaussian_filter(np.random.rand(5*resolution, 5*resolution, 5*resolution), sigma=0.3*resolution))
        # rho_0 = np.random.rand(5*resolution, 5*resolution, 5*resolution)


        init_proj = projection_loader("tanh_jax", threshold_value, 2, resolution)
        dose_filter = filter_loader("dose_conv", resolution, ConfigPrint.lp, psf)
        proj = projection_loader("ssp_jax", threshold_value, np.inf, resolution)

        rho_0_init = np.array(init_proj(rho_0))
        rho_filt = dose_filter(rho_0_init)
        result = np.array(rho_filt).squeeze()
        result = proj(result)
        result_bin = np.where(result > ConfigPrint.rho_th_GT, 1, 0)

        ax[i].imshow(result[:, :, rho_0.shape[2]//2], vmin=0, vmax=1)
        i += 1
    plt.show()
    return

if __name__ == "__main__":
    test(42342)