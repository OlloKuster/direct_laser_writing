import torch
import numpy as np

from utility.helper import f2param
from .config_print import ConfigPrint
from filtering.dose_model.DoseMSBPM import DoseMSBPMFull3D
from filtering.dose_model.utils_dose_sim import create_3d_psf_torch


def dose_filter_f(resolution):

    res_lat = 1 / resolution
    res_ax = 2 / resolution
    size_lat = int(np.ceil(9 / 10 * resolution))
    size_ax = int(np.ceil(17 / 10 * resolution))
    time_exposure = ConfigPrint.w0 / (ConfigPrint.vs / res_lat)

    psf_GT = create_3d_psf_torch(x_fwhm=torch.tensor(ConfigPrint.x_fwhm),
                                 y_fwhm=torch.tensor(ConfigPrint.y_fwhm),
                                 z_fwhm=torch.tensor(ConfigPrint.z_fwhm),
                                 rotation_xy=torch.tensor(ConfigPrint.rot_xy),
                                 rotation_xz=torch.tensor(ConfigPrint.rot_xz),
                                 rotation_yz=torch.tensor(ConfigPrint.rot_yz),
                                 astigmatism_xy=torch.tensor(ConfigPrint.astigmatism_xy),
                                 res_lat=torch.tensor(res_lat),
                                 res_ax=torch.tensor(res_ax),
                                 size_ax=size_ax,
                                 size_lat=size_lat)[None, None]

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

        return rho / torch.max(rho)

    return dose_filter
