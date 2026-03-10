import torch
import pyvista as pv
import numpy as np

from filtering.dose_model.config_print import ConfigPrint
from filtering.dose_model.utils_dose_sim import calc_laser_intensity

resolution = 30
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
print(psf.shape)
pv.plot(psf, cmap='magma')