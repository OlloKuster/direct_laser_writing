from dataclasses import dataclass

@dataclass
class ConfigSim:
    """
    Config the 3D lens, defines the required parameters which are used for the simulation.
    """
    resolution = 20
    simulation_domain_shape = (11, 11, 12)
    rho_shape = (5, 5, 5)
    currents_shape = (11, 11)
    sinks_shape = rho_shape
    wavelength = 1.55
    save_plot = True
    save_data = True
    epsilon = (1, 2.25)
    kappa = (1e-5, 1)
    resize_factor = 1

    dpml = 0.5
    location_currents = dpml + 1
    location_focal_spot = 6*resolution

    TARGET_EM = 20
    TARGET_MATERIAL = -1
    TARGET_VOID = 1.5
