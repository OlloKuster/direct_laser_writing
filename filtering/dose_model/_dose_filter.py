import matplotlib
import torch
import numpy as np

from utility.helper import f2param
from .config_print import ConfigPrint
from filtering.dose_model.DoseMSBPM import DoseMSBPMFull3D
from filtering.dose_model.utils_dose_sim import create_3d_psf_torch, calc_laser_intensity


def dose_filter_f(resolution):

    res_lat = 1 / resolution * 10**(-6) # hatching
    res_ax = 1 / resolution * 10**(-6) # slicing
    size_lat = int(np.ceil(9/10*resolution))
    size_ax = int(np.ceil(17/10*resolution))
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
                                  )[None, None] # calc_laser_intensity

    psf = psf_GT.detach().clone().requires_grad_(True)
    print_params = [ConfigPrint.sig_2_r.detach().clone().requires_grad_(True),
                    time_exposure.detach().clone().requires_grad_(True),
                    ConfigPrint.intensity_without_power.detach().clone().requires_grad_(True),
                    torch.tensor(ConfigPrint.correction_factor, requires_grad=True)]

    # import matplotlib.pyplot as plt
    # psf_plot = psf_GT.detach().cpu().numpy()
    # psf_plot = psf_plot.squeeze()
    # for i in range(psf_plot.shape[1]):
    #     plt.imshow(psf_plot[:, i])
    #     plt.colorbar()
    #     plt.savefig(f"/users/tfp/okuster/Pictures/psf/psf_{i}.png")
    #     plt.close()

    msbpm = DoseMSBPMFull3D(
        torch.tensor(ConfigPrint.rho_0_GT, device=ConfigPrint.device, requires_grad=True),
        torch.tensor(ConfigPrint.rho_th_GT, device=ConfigPrint.device, requires_grad=True),
        psf / resolution * 10,
        print_params,
        torch.tensor(ConfigPrint.nonlinearity, device=ConfigPrint.device, requires_grad=True),
        device=ConfigPrint.device
    )

    def dose_filter(rho_0):
        rho = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[ConfigPrint.lp]], device=ConfigPrint.device, requires_grad=True))

        return rho

    return dose_filter
