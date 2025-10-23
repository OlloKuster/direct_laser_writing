import jax
import numpy as np
import matplotlib.pyplot as plt

from filtering.dose_model.config_print import ConfigPrint
from problems.metalens.simulation.config_structure import ConfigSim
from projection._projection_loader import projection_loader


def test(seed):
    jax.config.update("jax_enable_x64", True)
    resolution = 10
    if seed == 0:
        rho_0 = np.ones((ConfigSim.rho_shape[0]*resolution,
                        ConfigSim.rho_shape[1]*resolution,
                        ConfigSim.rho_shape[2]*resolution)) * 0.5

    projection = projection_loader("tanh_jax", ConfigPrint.rho_th_GT, 16, resolution)

    rho_f = projection(rho_0)
    plt.imshow(rho_f.shape[0]//2)