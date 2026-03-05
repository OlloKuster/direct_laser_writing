import autograd
import jax
import jax.numpy as jnp
import numpy as np

from problems.mode_converter.simulation.objective import measure_mode_power, measure_mode_power_ag

rho_0 = np.ones((60, 60, 20))

jax_val, jax_grad = jax.value_and_grad(measure_mode_power)(jnp.array(rho_0))
print(np.mean(jax_grad))
ag_val, ag_grad = autograd.value_and_grad(measure_mode_power_ag)(rho_0)
print(np.mean(ag_grad))

print(np.linalg.norm(jax_grad - ag_grad))