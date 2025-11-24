import matplotlib
import torch
import numpy as np

from utility.helper import f2param
from .config_print import ConfigPrint
from filtering.dose_model.DoseMSBPM import DoseMSBPMFull3D
from filtering.dose_model.utils_dose_sim import create_3d_psf_torch, calc_laser_intensity


def dose_filter_f(resolution):
    res_lat = 1 / resolution * 10 ** (-6)  # hatching
    res_ax = 1 / resolution * 10 ** (-6)  # slicing
    time_exposure = ConfigPrint.w0 / (ConfigPrint.vs / res_lat)

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
    print_params = [ConfigPrint.sig_2_r.detach().clone().requires_grad_(True),
                    time_exposure.detach().clone().requires_grad_(True),
                    ConfigPrint.intensity_without_power.detach().clone().requires_grad_(True),
                    torch.tensor(ConfigPrint.correction_factor, requires_grad=True)]

    msbpm = DoseMSBPMFull3D(
        torch.tensor(ConfigPrint.rho_0_GT, device=ConfigPrint.device, requires_grad=True),
        psf,
        print_params,
        torch.tensor(ConfigPrint.nonlinearity, device=ConfigPrint.device, requires_grad=True),
        device=ConfigPrint.device
    )

    def dose_filter(rho_0):
        rho = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[ConfigPrint.lp]], device=ConfigPrint.device, requires_grad=True))

        return rho.squeeze()

    return dose_filter
