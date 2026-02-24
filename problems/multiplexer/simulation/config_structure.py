from dataclasses import dataclass
from tidy3d import C_0
import numpy as np

@dataclass
class ConfigSim:
    '''
    Config file for the tidy3d Simulation. Units are given in um.
    Propagation direction is in z-direction.
    '''

    eval_wvls = [1.25, 1.55, 1.75]
    eval_freqs = [C_0 / wvl for wvl in eval_wvls[::-1]]
    wavelength = np.mean(eval_wvls)
    freq0 = C_0 / wavelength

    fwidth = np.max(eval_freqs) - np.min(eval_freqs)
    run_time = 200 / fwidth
    rho_size = (30, 30, 1.5)
    thickness_substrate = 3
    buffer = 1 * wavelength

    buffer_side = 1
    buffer_top = 1

    number_wgs = 1

    wg_width = 2
    wg_height = 1
    wg_length = 3
    wg_dist = 3

    lx = wg_length + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = thickness_substrate + rho_size[2] + buffer

    pos_source = [-lx / 2 + 0.5, 0, -lz / 2 + thickness_substrate + wg_height / 2]
    size_source = [0, 3*wg_width, 3*wg_height]

    pos_monitor = [lx / 2 - 0.5, 0, -lz / 2 + thickness_substrate + wg_height / 2]
    size_monitor = [0, 1.5*wg_width, 1.5*wg_height]

    num_modes = 1

    refr_index = (1., 1.44, 1.53) # Air, SiO2, Polymer
    kappa = (1e-5, 1)  # Thermal conductivity
    min_feature_size = 0.5

    dl = 8
    nx = rho_size[0]*dl
    ny = rho_size[1]*dl
    nz = rho_size[2]*dl

    TARGET_MATERIAL = 0.8
    TARGET_VOID = 0.8

    resize_factor = 1

    p = 10

    sim_id = 0