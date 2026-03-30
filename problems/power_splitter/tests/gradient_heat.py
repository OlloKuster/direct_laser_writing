import jax
import numpy as np
import jax.numpy as jnp
import autograd

from problems.mode_converter.simulation.config_structure import ConfigSimMode
from problems.mode_converter.simulation.objective import objective_em_heat_f
from problems.mode_converter.simulation.simulation import make_sim_tidy


def test_gradient_heat():
    resolution = 10
    rho_0 = np.ones((ConfigSimMode.nx,
                     ConfigSimMode.ny,
                     ConfigSimMode.nz)) * 0.5
    objective = objective_em_heat_f((1, 1))
    val, grad = jax.value_and_grad(objective, has_aux=True)(rho_0)
    print(val)
    print(np.linalg.norm(grad))


if __name__ == "__main__":
    test_gradient_heat()
