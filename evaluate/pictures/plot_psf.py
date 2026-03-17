import torch
import pyvista as pv
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cmcrameri import cm

from filtering.dose_model.config_print import ConfigPrint
from filtering.dose_model.utils_dose_sim import calc_laser_intensity


resolution = 14


def gkernel(sigma):
    l = int(jnp.ceil(4.0 * sigma) + 1)
    ax = jnp.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy, zz = jnp.meshgrid(ax, ax, ax)

    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
    return kernel / jnp.sum(kernel)

gauss = gkernel(1/np.sqrt(3)*resolution)

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
                                  )

psf = np.clip(psf_GT.detach().cpu().numpy(), 5, 30)

plt.imshow(gauss[gauss.shape[0]//2].T, origin='lower', cmap=cm.lapaz_r, interpolation='spline36')
plt.show()
plt.axis('off')
plt.savefig("plots/gauss_filter/gauss.png")
plt.close()
plt.imshow(psf[psf.shape[0]//2].T, origin='lower', cmap=cm.lapaz_r, interpolation='spline36')
plt.axis('off')
plt.savefig("plots/dlw_filter/psf.png")
plt.close()


