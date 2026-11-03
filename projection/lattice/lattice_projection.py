import jax.numpy as jnp

from filtering.dose_model.config_print import ConfigPrint

def lattice_proj_jax_f(f2bin, resolution, offset=(0, 0)):
    """
    Generates the initial projection onto the writing lattice. The scan runs along x and is sampled at every voxel,
    the lines are m_y voxels apart in y and the layers m_z voxels apart in z.
    :param f2bin: Binarization of the exposure on the lattice, e.g. tanh_filter_jax_f(alpha, beta).
    :param grid_shape: Shape of the simulation grid.
    :param m_y: Voxels between two lines, see lattice_steps.
    :param m_z: Voxels between two layers, see lattice_steps.
    :param offset: Grid index of the first line and the first layer.
    :return: Lattice projection function, design [nx, n_lines, n_layers] -> written pattern on the grid [nx, ny, nz],
             which is zero between the lines and layers.
    """
    o_y, o_z = offset

    def lattice_proj(rho):
        """
        Binarizes the exposure of every lattice point and places it on the simulation grid.
        :param rho: Design variable on the writing lattice [nx, n_lines, n_layers].
        :return: Written pattern on the simulation grid [nx, ny, nz].
        """
        written = f2bin(rho)
        m_y, m_z = round(ConfigPrint.hatching * resolution), round(ConfigPrint.slicing * resolution)

        return written.at[:, o_y::m_y, o_z::m_z].set(0)

    return lattice_proj


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    f2bin = lambda x: x

    lattice_proj = lattice_proj_jax_f(f2bin,  20)

    rho = jnp.array(np.random.rand(50, 50, 50))

    proj = lattice_proj(rho)

    plt.imshow(proj[0].T, origin='lower')
    plt.show()
