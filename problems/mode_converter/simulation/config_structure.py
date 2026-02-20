from dataclasses import dataclass
from tidy3d import C_0

@dataclass
class ConfigSimMode:
    '''
    Config file for the tidy3d Simulation. Units are given in um.
    Propagation direction is in z-direction.
    '''
    wavelength = 1.
    freq0 = C_0 / wavelength
    fwidth = freq0 / 10
    run_time = 50 / fwidth

    rho_size = (5, 5, 5)
    thickness_substrate = 3
    buffer = 1 * wavelength

    buffer_side = 0.5
    buffer_top = 1

    wg_width = 4
    wg_height = 2
    wg_length = 3

    lx = wg_length + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = thickness_substrate + rho_size[2] + buffer

    pos_source = [-lx / 2 + 0.5, 0, -lz / 2 + thickness_substrate + wg_height / 2]
    size_source = [0, 3*wg_width, 3*wg_height]

    pos_monitor = [lx / 2 - 0.5, 0, -lz / 2 + thickness_substrate + wg_height / 2]
    size_monitor = [0, 3*wg_width, 3*wg_height]

    num_modes = 4

    refr_index = (1., 1.44, 1.53) # Air, SiO2, Polymer
    kappa = (1e-5, 1)  # Thermal conductivity
    min_feature_size = 0.5

    min_steps_per_wvl = 10
    dl = 10
    nx = rho_size[0]*dl
    ny = rho_size[1]*dl
    nz = rho_size[2]*dl

    TARGET_MATERIAL = 1.0
    TARGET_VOID = 1.0

    resize_factor = 1