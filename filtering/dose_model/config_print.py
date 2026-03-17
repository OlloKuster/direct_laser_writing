from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class ConfigPrint():
    """
    Parameters for the dlw-model. Most of them are based on experimental data.
    Threhold and initial values for rho are normalized and the inital power is 1.
    """
    device = 'cuda'

    power = 1.

    n_monomer = torch.tensor(1.516)

    h = torch.tensor(6.626e-34, device=device, dtype=torch.float64)
    c = torch.tensor(2.99792458e8, device=device, dtype=torch.float64)

    lam = torch.tensor(7.8e-7) # 6.3
    NA = torch.tensor(1.4) # 1.2
    t_p = torch.tensor(100e-15)
    r_p = torch.tensor(80e6)

    r_r = 4e-7
    r_z = 9e-7

    w0 = lam / (torch.pi * NA) * torch.sqrt(n_monomer ** 2 - NA ** 2)

    velocity = 0.01  # in nm/s, Schreibgeschwindigkeit
    VD = (velocity / 4) * 10 ** (-9)  # Voxeldistance m
    freq = 4000  # galvo frequency in Hz
    vs = VD * freq  # writing speed in m/s

    sig_2_r_GT = 3.98107171e-68 # 8e-711.12883789e-71
    sig_2_r_base = torch.tensor(sig_2_r_GT / (10 ** np.floor(np.log10(sig_2_r_GT))))
    sig_2_r_exp = torch.tensor(np.floor(np.log10(sig_2_r_GT)))
    # sig_2_r_base * 10 ** sig_2_r_exp

    x_fwhm = .415995  # um
    y_fwhm = .354342  # um
    z_fwhm = .963721  # um

    rot_xy = 24.7
    rot_xz = 10.5
    rot_yz = 6.3

    astigmatism_xy = .142847
    # size_lat = int(np.ceil(9 / 10 * resolution))
    # size_ax = int(np.ceil(17 / 10 * resolution))

    rho_0_GT = 1.
    # sig_2_r = torch.tensor(2.51188643e-67)
    sig_2_r = torch.tensor(sig_2_r_base * 10 ** sig_2_r_exp)
    # time_exposure = w0 / (vs / res_lat)
    intensity_without_power = (
            1 / (torch.pi * w0 ** 2 * c / lam * h * r_p * t_p) * 2 * torch.exp(
        torch.tensor(2)) / (torch.exp(torch.tensor(2)) - 1)
    )
    correction_factor = 1.
    nonlinearity = 2.0

    lp = 2*00204 # between 22-25-ish percent  ~0.05W
    rho_th_GT = 0.5 #1 / 6.2525