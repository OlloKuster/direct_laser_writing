from dataclasses import dataclass

import numpy as np
from tidy3d import C_0

@dataclass
class ConfigSim:
    '''
    Config file for the tidy3d Simulation. Units are given in um.
    Propagation direction is in z-direction.
    '''
    wavelength = 1.55
    freq0 = C_0 / wavelength
    fwidth = freq0 / 10
    f_eval = np.linspace(freq0-fwidth, freq0+fwidth, 11)
    run_time = 50 / fwidth

    rho_size = (11, 6, 6)
    thickness_substrate = 2
    buffer = 1 * wavelength

    buffer_side = 0.5
    buffer_top = 1

    wg_width = 1
    wg_height = 1
    wg_length = 3

    lx = wg_length + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = rho_size[2] + 2*buffer

    pos_source = [-lx / 2 + 0.5, 0, 0]
    size_source = [0, 2*wg_width, 2*wg_width]

    pos_monitor = [lx / 2 - 0.5, rho_size[1]/4, rho_size[2]/4]
    size_monitor = [0, 2*wg_width, 2*wg_width]

    num_modes = 1

    refr_index = (1., 1.53) # Air, SiO2, Polymer
    kappa = (1e-5, 1)  # Thermal conductivity
    min_feature_size = 0.5

    min_steps_per_wvl = 8
    dl = 14
    nx = rho_size[0]*dl
    ny = rho_size[1]*dl
    nz = rho_size[2]*dl

    TARGET_MATERIAL = -0.8
    TARGET_VOID = -0.8

    resize_factor = 1

    cur_it = 0