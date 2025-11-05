import numpy as np
import jax.numpy as jnp
import scipy.ndimage
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pyvista as pv

from filtering._filter_loader import filter_loader
from filtering.dose_model.config_print import ConfigPrint
from filtering.dose_model.utils_dose_sim import calc_laser_intensity
from filtering.gaussian_filter._gaussian_filter import conic_filter_jax_f


def test(seed):
    np.random.seed(seed)
    resolution = 20
    res_lat = 1 / resolution * 10 ** (-6)  # hatching
    res_ax = 1 / resolution * 10 ** (-6)  # slicing

    psf_GT = calc_laser_intensity(lam=torch.tensor(ConfigPrint.lam),
                                  NA=torch.tensor(ConfigPrint.NA),
                                  M=torch.tensor(64.),
                                  r_r=torch.tensor(ConfigPrint.r_r),
                                  r_z=torch.tensor(ConfigPrint.r_z),
                                  res_ax=res_ax,
                                  res_lat=res_lat,
                                  n_monomer=ConfigPrint.n_monomer,
                                  torch_device='cuda',
                                  )[None, None]  # calc_laser_intensity

    psf = psf_GT.detach().clone().requires_grad_(True)

    psf = psf.detach().cpu().numpy().squeeze()

    p = pv.Plotter()
    data = pv.wrap(np.clip(psf, 3, 1000))
    p.add_volume(data, cmap='magma')
    # p.show()

    def gkernel(radius):
        l = int(2 * radius + 1)
        ax = jnp.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy, zz = jnp.meshgrid(ax, ax, ax)

        kernel = jnp.maximum(radius - jnp.sqrt(xx ** 2 + yy ** 2 + zz ** 2), jnp.zeros_like(xx))
        return kernel / jnp.sum(kernel)

    cone = gkernel(resolution / np.sqrt(3))
    p = pv.Plotter()
    data = pv.wrap(np.array(cone))
    p.add_volume(data, cmap='magma')
    # p.show()


    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cone[cone.shape[0]//2].T, origin='lower', interpolation='spline36')
    axs[1].imshow(psf[psf.shape[0]//2].T, origin='lower', interpolation='spline36')
    axs[0].set_xlabel(r"y in $\mathregular{\mu}$m")
    axs[1].set_xlabel(r"y in $\mathregular{\mu}$m")
    axs[0].set_ylabel(r"z in $\mathregular{\mu}$m")
    axs[1].set_ylabel(r"z in $\mathregular{\mu}$m")
    plt.show()

if __name__ == "__main__":
    test(152)
