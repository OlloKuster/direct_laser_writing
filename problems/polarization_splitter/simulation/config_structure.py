from dataclasses import dataclass
from tidy3d import C_0

@dataclass
class ConfigSim:
    '''
    Config file for the tidy3d Simulation. Units are given in um.
    Propagation direction is in z-direction.
    '''
    p = 5

    wavelength = 1.55
    freq0 = C_0 / wavelength
    fwidth = freq0 / 10
    run_time = 50 / fwidth

    rho_size = (11, 7, 4)
    thickness_substrate = 2
    buffer = 1 * wavelength

    buffer_side = 0.5
    buffer_top = 1

    wg_width = 1
    wg_height = 0.5
    wg_init_width = 1
    wg_init_height = 0.5
    wg_length = 3
    wg_spacing = 8

    lx = wg_length + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = thickness_substrate + rho_size[2] + buffer

    pos_source = [-lx / 2 + 0.5, -(rho_size[1] - wg_init_width) / 2 + 2*buffer_side, 0]
    size_source = [0, 3*wg_init_width, 3*wg_init_width]

    pos_monitor_te = [lx / 2 - wg_length + 0.5, -(rho_size[1] - wg_width) / 2 + 2*buffer_side, 0]
    pos_monitor_tm = [lx / 2 - wg_length + 0.5, (rho_size[1] - wg_width) / 2 - 2*buffer_side, 0]
    size_monitor = [0, 3*wg_width, 3*wg_width]
    size_monitor_te = [0, 3*wg_width, 3*wg_height]
    size_monitor_tm = [0, 3*wg_height, 3*wg_width]

    num_modes = 2

    refr_index = (1., 1., 1.53) # Air, SiO2, Polymer
    kappa = (1e-5, 1)  # Thermal conductivity
    min_feature_size = 0.5

    min_steps_per_wvl = 14
    dl = 14
    nx = rho_size[0]*dl
    ny = rho_size[1]*dl
    nz = rho_size[2]*dl

    TARGET_MATERIAL = 1.3
    TARGET_VOID = 1.0

    resize_factor = 1

    cur_it = 0