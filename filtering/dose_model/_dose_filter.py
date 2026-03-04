import torch

from .config_print import ConfigPrint
from filtering.dose_model.DoseMSBPM import DoseMSBPMFull3D
from filtering.dose_model.utils_dose_sim import calc_laser_intensity


def dose_filter_f(resolution, lp=ConfigPrint.lp):
    """
    Creases the function for the dlw-model filter, given a resolution.
    :param resolution: Resolution of the simulation [px/um].
    :return: Filter function, with density as its input.
    """
    res_lat = 1/resolution * 10 ** (-6)  # hatching
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

    # import matplotlib.pyplot as plt
    # psf_plot = psf_GT.detach().cpu().numpy().squeeze()
    # plt.imshow(psf_plot[psf_plot.shape[0]//2].T, origin='lower')
    # plt.show()

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
        """
        The dose filter function, converts the regular density into a power accumulated density which can be
        converted into a printed structure using a threshold value.
        :param rho_0: Regular TopOpt density.
        :return: Accumulated power as a "density".
        """
        rho = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[lp]], device=ConfigPrint.device, requires_grad=True))

        return rho.squeeze()

    return dose_filter
def dose_filter_robust_f(resolution, deviation):
    """
    Creases the function for the dlw-model filter, given a resolution.
    :param resolution: Resolution of the simulation [px/um].
    :return: Filter function, with density as its input.
    """
    res_lat = 1/resolution * 10 ** (-6)  # hatching
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

    # import matplotlib.pyplot as plt
    # psf_plot = psf_GT.detach().cpu().numpy().squeeze()
    # plt.imshow(psf_plot[psf_plot.shape[0]//2].T, origin='lower')
    # plt.show()

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

    def dose_filter_robust(rho_0):
        """
        The dose filter function, converts the regular density into a power accumulated density which can be
        converted into a printed structure using a threshold value.
        :param rho_0: Regular TopOpt density.
        :return: Accumulated power as a "density".
        """
        rho_eroded = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[(1.-deviation) *ConfigPrint.lp]], device=ConfigPrint.device, requires_grad=True))
        rho_normal = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[ConfigPrint.lp]], device=ConfigPrint.device, requires_grad=True))
        rho_dilated = msbpm(rho_0 * torch.tensor(ConfigPrint.power, device='cuda', requires_grad=True),
                    torch.tensor([[(1. + deviation) * ConfigPrint.lp]], device=ConfigPrint.device, requires_grad=True))

        return torch.stack((rho_eroded.squeeze(), rho_normal.squeeze(), rho_dilated.squeeze()))

    return dose_filter_robust
