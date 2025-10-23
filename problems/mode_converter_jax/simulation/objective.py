import jax.numpy as jnp
from jax.debug import print as jprint

from problems.mode_converter_jax.simulation.config_structure import ConfigSim
from problems.mode_converter_jax.simulation.simulation import em_simulation
from utility.helper import softplus


def objective_em_f(currents, resolution, target_field):
    def objective_em(rho):
        monitor_pos = (int(jnp.ceil(ConfigSim.monitor_pos[0]*resolution)),
                       int(jnp.ceil(ConfigSim.monitor_pos[1]*resolution)),
                       int(jnp.ceil(ConfigSim.monitor_pos[2]*resolution)))
        monitor_size = (1,
                        int(jnp.ceil(2*ConfigSim.wg_width*resolution)),
                        int(jnp.ceil(2*ConfigSim.wg_width*resolution)))
        E, eps = em_simulation(rho, currents, resolution)
        field_monitor = E[2][monitor_pos[0],
                        monitor_pos[1]-monitor_size[1]//2:monitor_pos[1]+monitor_size[1]//2,
                        monitor_pos[2]-monitor_size[2]//2:monitor_pos[2]+monitor_size[2]//2]
        target_field_monitor = target_field[0,
                        monitor_pos[1]-monitor_size[1]//2:monitor_pos[1]+monitor_size[1]//2,
                        monitor_pos[2]-monitor_size[2]//2:monitor_pos[2]+monitor_size[2]//2]
        return jnp.abs(jnp.sum(jnp.conj(field_monitor) * target_field_monitor))
    return objective_em
