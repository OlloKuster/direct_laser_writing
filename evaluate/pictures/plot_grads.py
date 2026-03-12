import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import h5py
from cmcrameri import cm

from problems.metalens.simulation._objective_loader import objective_loader
from problems.metalens.simulation.config_structure import ConfigSim
from projection._projection_loader import projection_loader


def test(seed, path):
    jax.config.update("jax_enable_x64", True)

    resolution = 14
    lr = 2e5
    if seed == 0:
        rho_0 = jnp.ones((ConfigSim.rho_shape[0] * resolution,
                          ConfigSim.rho_shape[1] * resolution,
                          ConfigSim.rho_shape[2] * resolution)) * 0.5

    if path != None:
        f = h5py.File(path)
        grp = f["grads"]
        grad_0 = grp["g"][:]
        f.close()
    else:
        currents = jnp.ones((ConfigSim.currents_shape[0] * resolution, ConfigSim.currents_shape[1] * resolution, 1),
                            jnp.complex128)

        objective_em = objective_loader("em_only", currents, resolution, 1, 1, 1)
        fom = objective_em(rho_0)
        value, grad_0 = jax.value_and_grad(objective_em, has_aux=True)(rho_0)

    grad = np.array(grad_0)
    with h5py.File("plots/grad.h5", 'w') as f:
        grp = f.create_group("grads")
        grp.create_dataset("g", data=grad)
        f.close()
    grad = np.concatenate((grad, np.flip(grad, axis=0)), axis=0)
    grad = np.concatenate((grad, np.flip(grad, axis=1)), axis=1)
    rho = np.concatenate((rho_0, np.flip(rho_0, axis=0)), axis=0)
    rho = np.concatenate((rho, np.flip(rho, axis=1)), axis=1)


    plt.imshow(rho[rho.shape[0]//2-5].T, origin='lower', cmap='binary', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig("plots/rho_0.png")
    plt.close()
    plt.imshow(grad[grad.shape[0]//2-5].T, origin='lower', alpha=1, cmap=cm.cork)
    plt.axis('off')
    plt.savefig("plots/grads.png")
    plt.close()

    rho_1 = rho_0 - lr * grad_0



    rho1 = np.concatenate((rho_1, np.flip(rho_1, axis=0)), axis=0)
    rho1 = np.clip(np.concatenate((rho1, np.flip(rho1, axis=1)), axis=1), 0, 1)

    plt.imshow(rho1[rho1.shape[0]//2-5].T, origin='lower', cmap='binary', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig("plots/rho_1.png")
    plt.close()


if __name__ == "__main__":
    test(0, "plots/grad.h5")
