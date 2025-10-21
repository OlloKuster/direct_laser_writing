from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class ConfigPrint():
    device = 'cuda'

    power = 1.

    n_monomer = torch.tensor(1.5**2)

    h = torch.tensor(6.626e-34, device=device, dtype=torch.float64)
    c = torch.tensor(2.99792458e8, device=device, dtype=torch.float64)

    lam = torch.tensor(7.8e-7) # 6.3
    NA = torch.tensor(1.4) # 1.2
    t_p = torch.tensor(100e-15)
    r_p = torch.tensor(80e6)

    w0 = lam / (torch.pi * NA) * torch.sqrt(n_monomer ** 2 - NA ** 2)

    velocity = 100  # in nm/s, Schreibgeschwindigkeit
    VD = (velocity / 4) * 10 ** (-9)  # Voxeldistance m
    freq = 4000  # galvo frequency in Hz
    vs = VD * freq  # writing speed in m/s

    sig_2_r_GT = 2.01958975e-71  # 8e-711.12883789e-71
    sig_2_r_base = torch.tensor(sig_2_r_GT / (10 ** np.floor(np.log10(sig_2_r_GT))))
    sig_2_r_exp = torch.tensor(np.floor(np.log10(sig_2_r_GT)))
    sig_2_r_base * 10 ** sig_2_r_exp

    x_fwhm = .415995  # um
    y_fwhm = .354342  # um
    z_fwhm = .963721  # um

    astigmatism_xy = .142847
    # size_lat = int(np.ceil(9 / 10 * resolution))
    # size_ax = int(np.ceil(17 / 10 * resolution))

    rho_0_GT = 0.07189474
    sig_2_r = sig_2_r_base * 10 ** sig_2_r_exp
    # time_exposure = w0 / (vs / res_lat)
    intensity_without_power = (
            1 / (torch.pi * w0 ** 2 * c / lam * h * r_p * t_p) * 2 * torch.exp(
        torch.tensor(2)) / (torch.exp(torch.tensor(2)) - 1)
    )
    correction_factor = 1.
    nonlinearity = 2.0

    lp = 0.05*0.2  # between 22-25-ish percent  ~0.05W
    rho_th_GT = 0.01